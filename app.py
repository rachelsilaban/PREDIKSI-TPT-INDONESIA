from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from model_utils import train_lstm
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
DATASET = "dataset.csv"

# LOAD & CLEAN DATASET
def load_and_clean():
    df_raw = pd.read_csv(DATASET, skiprows=1)
    df_raw.rename(columns={df_raw.columns[0]: "provinsi"}, inplace=True)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("unnamed", case=False)]
    df_raw = df_raw.dropna(axis=1, how="all")
    
    df_long = df_raw.melt(id_vars="provinsi", var_name="tahun", value_name="tpt")
    df_long["tahun"] = pd.to_numeric(df_long["tahun"], errors="coerce")
    df_long["tpt"] = pd.to_numeric(df_long["tpt"], errors="coerce")
    df_long = df_long.dropna()
    return df_long

df_long = load_and_clean()
provinsi_list = sorted(df_long["provinsi"].unique())

# FLASK ROUTE
@app.route("/", methods=["GET","POST"])
def index():
    result = None
    chart = None
    shap_img = None
    shap_error = None

    if request.method == "POST":
        prov = request.form.get("provinsi")
        steps = int(request.form.get("steps", 5))
        show_trend = request.form.get("show_trend") == "on"
        show_chart = request.form.get("show_chart") == "on"

        if prov:
            df_p = df_long[df_long["provinsi"] == prov].sort_values("tahun")
            series = df_p["tpt"].values
            years = df_p["tahun"].tolist()

            # TRAIN LSTM
            future, mse, rmse, mae, model = train_lstm(series, steps=steps)
            future_years = list(range(years[-1] + 1, years[-1] + 1 + len(future)))

            result = {
                "provinsi": prov,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "tahun": years + future_years,
                "tpt": series.tolist() + future
            }

            # VISUALISASI
            if show_trend or show_chart:
                fig, ax = plt.subplots(figsize=(8, 5))
                if show_trend:
                    ax.plot(years, series, marker="o", label="Aktual")
                    ax.plot(future_years, future, marker="x", linestyle="--", label="Prediksi")
                if show_chart and not show_trend:
                    ax.bar(years + future_years, series.tolist() + future, color="skyblue")

                ax.set_title(f"TPT {prov}")
                ax.set_xlabel("Tahun")
                ax.set_ylabel("TPT (%)")
                ax.legend()
                ax.grid(True)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                chart = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)

            # SHAP using KernelExplainer
            try:
                scaler = MinMaxScaler()
                series_scaled = scaler.fit_transform(series.reshape(-1,1))

                def model_predict(X_input):
                    X_input = np.array(X_input).reshape(-1,1,1)
                    pred = model.predict(X_input, verbose=0)
                    return pred.flatten()

                X_background = series_scaled[-20:].reshape(-1,1)
                explainer = shap.KernelExplainer(model_predict, X_background)

                X_test = series_scaled[-5:].reshape(-1,1)
                shap_values = explainer.shap_values(X_test, nsamples=50)

                # SHAP plot
                plt.figure(figsize=(8,5))
                shap.summary_plot(shap_values, X_test, show=False)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                shap_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close()

            except Exception as e:
                shap_error = str(e)

    return render_template(
        "index.html",
        provinsi_list=provinsi_list,
        result=result,
        chart=chart,
        shap_img=shap_img,
        shap_error=shap_error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
