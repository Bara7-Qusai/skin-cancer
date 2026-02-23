resultDiv.innerHTML = `
    <div style="margin-bottom:10px;">
        <span class="label">${data.prediction}</span>
    </div>
    <div style="margin-bottom:10px;">
        Confidence:
        <span class="conf">${data.prediction_probability}</span>
    </div>
    <div style="margin-top:15px;">
        <img src="data:image/png;base64,${data.heatmap}"
             style="max-width:100%; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,.15);" />
    </div>
`;
