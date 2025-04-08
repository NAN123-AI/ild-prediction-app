from flask import Flask, request, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 加载模型和标准化器
gbdt_model = load('gbdt_model.joblib')
scaler = load('scaler.joblib')

# 定义需要的特征（新增ALT和血沉两个特征）
FEATURES = ['ALT', '血沉', '白蛋白', '抗合成酶抗体阳性', '血红蛋白', '甘油三酯']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # 获取用户输入并验证
            alt = float(request.form['alt'])
            erythrocyte_sedimentation = float(request.form['erythrocyte_sedimentation'])

            # 验证血沉不能为0
            if erythrocyte_sedimentation == 0:
                raise ValueError("血沉值不能为0")

            input_data = {
                'ALT': alt,
                '血沉': erythrocyte_sedimentation,
                'ALT_÷_血沉': alt / erythrocyte_sedimentation,  # 保持原有特征计算
                '白蛋白': float(request.form['albumin']),
                '抗合成酶抗体阳性': int(request.form['antibody']),
                '血红蛋白': float(request.form['hemoglobin']),
                '甘油三酯': float(request.form['triglyceride'])
            }

            # 验证抗体输入是否为0或1
            if input_data['抗合成酶抗体阳性'] not in {0, 1}:
                raise ValueError("抗体值必须为0或1")

            # 转换为DataFrame并标准化（使用原始模型需要的特征）
            X_new = pd.DataFrame([input_data])[['ALT_÷_血沉', '白蛋白', '抗合成酶抗体阳性', '血红蛋白', '甘油三酯']]
            X_new_scaled = scaler.transform(X_new)

            # 预测概率并格式化结果
            prob = gbdt_model.predict_proba(X_new_scaled)[0][1]
            if prob >= 0.5:
                result = f"ILD分级为1级"
            else:
                result = f"ILD分级为0级"

            return render_template('index.html', result=result, form_data=input_data)

        except Exception as e:
            error_msg = f"输入错误: {str(e)}"
            return render_template('index.html', error=error_msg, form_data=request.form)

    return render_template('index.html', result=None, form_data=None)


if __name__ == '__main__':
    app.run(debug=True)