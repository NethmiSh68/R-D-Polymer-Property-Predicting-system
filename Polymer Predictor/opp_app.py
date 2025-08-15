"""


Run:
    python opp_app.py
Open:
    http://localhost:8001
"""

import os
import io
import csv
import base64
import urllib.parse
from http import server
from http import HTTPStatus

import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.neighbors import NearestNeighbors

# Matplotlib for CI bar chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
N_BITS = 2048
RADIUS = 2

# -----------------------------
# Helpers (features & images)
# -----------------------------
def smiles_to_morgan_fp(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr

def mol_png_base64(smiles: str, size=(360, 260)) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def mols_grid_png_base64(smiles_list: list[str], mols_per_row=3, sub_size=(240, 180)) -> str | None:
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        AllChem.Compute2DCoords(m)
        mols.append(m)
    if not mols:
        return None
    grid = Draw.MolsToGridImage(mols, molsPerRow=min(mols_per_row, len(mols)), subImgSize=sub_size)
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def bar_chart_png_base64(preds: dict, intervals: dict) -> str:
    props = PROPERTIES
    means = [preds[p] for p in props]
    yerr = [
        [max(0.0, preds[p] - intervals[p][0]) for p in props],
        [max(0.0, intervals[p][1] - preds[p]) for p in props],
    ]
    x = np.arange(len(props))
    fig = plt.figure(figsize=(6.8, 3.2), dpi=150)
    plt.bar(x, means, yerr=yerr, capsize=4)
    plt.xticks(x, props)
    plt.ylabel("Prediction")
    plt.title("Predicted properties with 95% CI")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------------
# Model file discovery
# -----------------------------
def find_model_path(model_dir: str, prop: str) -> str | None:
    """
    Accept any of these patterns per property (case-sensitive):
      {prop}_rf_model.pkl
      {prop}_rf.pkl
      {prop}_rf_1024.pkl
    """
    candidates = [
        f"{prop}_rf_model.pkl",
        f"{prop}_rf.pkl",
        f"{prop}_rf_1024.pkl",
    ]
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    return None

# -----------------------------
# Core predictor
# -----------------------------
class PolymerPredictor:
    def __init__(self, model_dir: str, feature_path: str, train_csv_path: str):
        # Load models (with flexible filenames)
        self.models = {}
        missing = []
        for prop in PROPERTIES:
            path = find_model_path(model_dir, prop)
            if path is None:
                missing.append(prop)
            else:
                with open(path, "rb") as f:
                    self.models[prop] = pickle.load(f)
        if missing:
            available = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
            msg = (
                "\n[Model loader] Missing model files for properties: "
                + ", ".join(missing)
                + "\nLooked for any of: *_rf_model.pkl, *_rf.pkl, *_rf_1024.pkl"
                + f"\nModel directory: {model_dir}"
                + f"\nAvailable files: {available}"
            )
            raise FileNotFoundError(msg)

        # Load training data for neighbours
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Missing training CSV: {train_csv_path}")

        self.X_train_bool = np.load(feature_path).astype(bool)
        self.train_df = pd.read_csv(train_csv_path)

        # Fit Jaccard NN on boolean Morgan bits
        self.neigh = NearestNeighbors(metric="jaccard", n_neighbors=3)
        self.neigh.fit(self.X_train_bool)

    def predict_single(self, smiles: str):
        fp = smiles_to_morgan_fp(smiles).reshape(1, -1)
        preds, intervals = {}, {}
        for prop, model in self.models.items():
            tree_vals = np.array([t.predict(fp)[0] for t in model.estimators_], dtype=float)
            mean = float(tree_vals.mean())
            std = float(tree_vals.std())
            preds[prop] = mean
            intervals[prop] = (mean - 1.96 * std, mean + 1.96 * std)
        return preds, intervals

    def nearest_neighbours(self, smiles: str, k: int = 3) -> pd.DataFrame:
        fp = smiles_to_morgan_fp(smiles).astype(bool)
        distances, idx = self.neigh.kneighbors([fp], n_neighbors=k)
        out = self.train_df.loc[idx[0], ["id", "SMILES"] + PROPERTIES].reset_index(drop=True)
        out.insert(1, "Similarity", 1.0 - distances[0])
        return out

    def predict_batch(self, smiles_list: list[str]) -> pd.DataFrame:
        rows = []
        for smi in smiles_list:
            preds, intervals = self.predict_single(smi)
            rec = {"SMILES": smi}
            for p in PROPERTIES:
                rec[f"{p}_pred"] = preds[p]
                rec[f"{p}_ci_low"] = intervals[p][0]
                rec[f"{p}_ci_high"] = intervals[p][1]
            rows.append(rec)
        return pd.DataFrame(rows)

# -----------------------------
# HTTP Handler
# -----------------------------
class RequestHandler(server.BaseHTTPRequestHandler):
    predictor = None  # set in initialize()

    def _set_headers(self, status=HTTPStatus.OK, content_type='text/html; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.end_headers()

    @classmethod
    def initialize(cls, model_dir: str, feature_path: str, train_csv_path: str):
        cls.predictor = PolymerPredictor(model_dir, feature_path, train_csv_path)

    def do_GET(self):
        html = self.render_form()
        self._set_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        ctype = self.headers.get('Content-Type', '')
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        result_html = ''
        try:
            if ctype.startswith('application/x-www-form-urlencoded'):
                params = urllib.parse.parse_qs(body.decode('utf-8'))
                smiles = params.get('smiles', [''])[0].strip()
                k = int(params.get('k', ['3'])[0])
                if smiles:
                    result_html = self.render_single_result(smiles, k)
                else:
                    result_html = '<p class="error">Please enter a valid SMILES.</p>'
            elif ctype.startswith('multipart/form-data'):
                boundary = ctype.split('boundary=')[-1].encode('utf-8')
                parts = body.split(b'--' + boundary)
                smiles = ''
                csv_data = None
                k = 3
                for part in parts:
                    if b'Content-Disposition' in part and b'\r\n\r\n' in part:
                        header, data = part.split(b'\r\n\r\n', 1)
                        header_str = header.decode('utf-8')
                        if 'name="smiles"' in header_str:
                            smiles = data.decode('utf-8').strip('\r\n')
                        elif 'name="k"' in header_str:
                            val = data.decode('utf-8').strip('\r\n')
                            try:
                                k = int(val)
                            except:
                                k = 3
                        elif 'name="csvfile"' in header_str:
                            csv_data = data.rstrip(b'\r\n')
                if csv_data:
                    result_html = self.render_batch_result(csv_data)
                elif smiles:
                    result_html = self.render_single_result(smiles, k)
                else:
                    result_html = '<p class="error">Provide a SMILES or upload a CSV.</p>'
            else:
                result_html = '<p class="error">Unsupported content type.</p>'
        except Exception as e:
            result_html = f'<p class="error">Error: {e}</p>'

        html = self.render_form(result_html)
        self._set_headers()
        self.wfile.write(html.encode("utf-8"))

    # -------------------------
    # Rendering helpers (HTML)
    # -------------------------
    def render_form(self, result_html: str = '') -> str:
        styles = """
        <style>
          :root { --bg:#0b1020; --panel:#11162a; --accent:#5aa9e6; --muted:#cbd5e1; --text:#e2e8f0; }
          * { box-sizing: border-box; font-family: Inter, ui-sans-serif, system-ui, -apple-system; }
          body { margin:0; background: linear-gradient(180deg, #0b1020 0%, #141a33 100%); color: var(--text); }
          .wrap { max-width: 1080px; margin: 0 auto; padding: 24px; }
          .brand { display:flex; align-items:baseline; gap:12px; margin-bottom: 16px; }
          .brand h1 { font-size: 28px; margin:0; }
          .brand .sub { color: var(--muted); font-size: 14px; }
          .panel { background: var(--panel); border:1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
          .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
          .field { display:flex; flex-direction:column; gap:6px; margin-bottom:10px; }
          textarea,input[type="number"],input[type="file"] { background:#0c1226; color:var(--text); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:10px; }
          label { color:#cbd5e1; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
          button { background:var(--accent); color:#0a0f1f; border:none; border-radius:10px; padding:10px 14px; font-weight:600; cursor:pointer; }
          button:hover { filter: brightness(1.05); }
          .card { background: #0d1530; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 12px; }
          .muted { color: var(--muted); font-size: 13px; }
          table { width:100%; border-collapse: collapse; }
          th, td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.08); vertical-align: top; }
          th { text-align:left; color:#a6b0cf; }
          .imgbox { display:flex; gap:14px; flex-wrap:wrap; }
          @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
          .error { color:#ff6b6b; font-weight:600; }
        </style>
        """
        form = """
        <div class="panel">
          <div class="grid">
            <form method="POST" enctype="application/x-www-form-urlencoded">
              <div class="field">
                <label for="smiles">SMILES (single)</label>
                <textarea id="smiles" name="smiles" rows="3" placeholder="Paste SMILES here..."></textarea>
              </div>
              <div class="field">
                <label for="k">Nearest Neighbours (k)</label>
                <input id="k" name="k" type="number" value="3" min="1" max="6"/>
              </div>
              <button type="submit">Predict</button>
              <p class="muted">Returns 5 properties + 95% CI, 2D depiction, and nearest neighbours.</p>
            </form>
            <form method="POST" enctype="multipart/form-data">
              <div class="field">
                <label for="csvfile">Batch CSV (must contain column <code>SMILES</code>)</label>
                <input id="csvfile" name="csvfile" type="file" accept=".csv"/>
              </div>
              <button type="submit">Run Batch</button>
              <p class="muted">Uploads up to thousands of candidates and gives a downloadable CSV.</p>
            </form>
          </div>
        </div>
        """
        header = f"""
        <!doctype html>
        <html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>{styles}</head>
        <body><div class="wrap">
            <div class="brand">
              <h1>R&amp;D Polymer Property Predictor</h1>
              <span class="sub">Deep learning mini-project · SMILES → properties, CI & neighbours</span>
            </div>
            {form}
            {result_html}
        </div></body></html>
        """
        return header

    def render_single_result(self, smiles: str, k: int) -> str:
        preds, intervals = self.predictor.predict_single(smiles)
        # Prediction table
        rows = []
        for p in PROPERTIES:
            lo, hi = intervals[p]
            rows.append(f"<tr><th>{p}</th><td>{preds[p]:.4f}</td><td>[{lo:.4f}, {hi:.4f}]</td></tr>")
        pred_table = f"""
        <div class="panel" style="margin-top:16px">
          <div class="grid">
            <div class="card">
              <h3>Prediction</h3>
              <table><tr><th>Property</th><th>Value</th><th>95% CI</th></tr>{''.join(rows)}</table>
            </div>
            <div class="card">
              <h3>Visualizations</h3>
              <div class="imgbox">
                {self._mol_img_tag(smiles)}
                {self._pred_chart_img_tag(preds, intervals)}
              </div>
              <p class="muted">Left: 2D depiction · Right: bar chart with 95% CI</p>
            </div>
          </div>
        </div>
        """

        # Nearest neighbours
        nn = self.predictor.nearest_neighbours(smiles, k=max(1, min(6, k)))
        nn_rows = []
        for _, row in nn.iterrows():
            cells = [f"<td>{row['id']}</td>", f"<td>{row['Similarity']:.3f}</td>", f"<td><code>{row['SMILES']}</code></td>"]
            for p in PROPERTIES:
                val = row[p]
                cells.append(f"<td>{'' if pd.isna(val) else f'{val:.4f}'}</td>")
            nn_rows.append("<tr>" + "".join(cells) + "</tr>")

        grid_img = mols_grid_png_base64(nn["SMILES"].tolist())
        grid_block = f'<img src="{grid_img}" alt="Nearest neighbours" style="max-width:100%;border-radius:8px"/>' if grid_img else ""

        nn_table = f"""
        <div class="panel" style="margin-top:12px">
          <h3>Nearest Neighbours</h3>
          <div class="imgbox">{grid_block}</div>
          <table style="margin-top:10px">
            <tr><th>id</th><th>Similarity</th><th>SMILES</th>{''.join([f'<th>{p}</th>' for p in PROPERTIES])}</tr>
            {''.join(nn_rows)}
          </table>
        </div>
        """
        return pred_table + nn_table

    def render_batch_result(self, csv_bytes: bytes) -> str:
        try:
            text = csv_bytes.decode("utf-8")
            reader = csv.reader(io.StringIO(text))
            header = next(reader, None)
            if header is None:
                return '<p class="error">Empty CSV.</p>'
            lower = [h.lower() for h in header]
            if "smiles" not in lower:
                return '<p class="error">CSV must contain a column named SMILES.</p>'
            smiles_idx = lower.index("smiles")
            smiles_list = [row[smiles_idx] for row in reader if len(row) > smiles_idx]
            res_df = self.predictor.predict_batch(smiles_list)
            preview = res_df.head(15).to_html(index=False).replace("NaN", "")
            out = io.StringIO()
            res_df.to_csv(out, index=False)
            b64 = base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")
            link = f'<a download="predictions.csv" href="data:text/csv;base64,{b64}">Download full CSV</a>'
            return f"""
            <div class="panel" style="margin-top:16px">
              <h3>Batch Predictions</h3>
              <div class="card">{link}</div>
              <div style="margin-top:10px">{preview}</div>
            </div>
            """
        except Exception as e:
            return f'<p class="error">Could not process CSV: {e}</p>'

    def _mol_img_tag(self, smiles: str) -> str:
        img = mol_png_base64(smiles)
        if not img:
            return '<div class="card">Invalid SMILES</div>'
        return f'<img src="{img}" alt="Molecule" style="border-radius:8px;max-width:100%"/>'

    def _pred_chart_img_tag(self, preds: dict, intervals: dict) -> str:
        chart = bar_chart_png_base64(preds, intervals)
        return f'<img src="{chart}" alt="Prediction chart" style="border-radius:8px;max-width:100%"/>'

# -----------------------------
# Server bootstrap
# -----------------------------
def run_server(model_dir: str, feature_path: str, train_csv_path: str, host: str = '127.0.0.1', port: int = 8000):
    RequestHandler.initialize(model_dir, feature_path, train_csv_path)
    with server.HTTPServer((host, port), RequestHandler) as httpd:
        print(f"Serving on http://{host}:{port} ...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    # If your models are in ./Trained, change the next line:
    MODEL_DIR = os.path.join(BASE, "opp_models_simpler")
    FEATURE_PATH = os.path.join(BASE, "train_morgan_features_2048.npy")
    TRAIN_CSV = os.path.join(BASE, "train_full.csv")
    run_server(MODEL_DIR, FEATURE_PATH, TRAIN_CSV, host="127.0.0.1", port=8001)
