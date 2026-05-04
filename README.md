---
title: SVM Explorer
emoji: 🧠
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 SVM Explorer

یک ابزار تعاملی برای یادگیری و آزمایش **Support Vector Machine (SVM)** ساخته‌شده با Streamlit.

## ✨ امکانات

| بخش | توضیح |
|-----|-------|
| 📖 آموزش مفاهیم | توضیح بصری SVM، Margin، Kernel Trick |
| 🎯 Decision Boundary | نمایش زنده مرز تصمیم با تنظیم hyperparameter |
| 🔬 مقایسه Kernel‌ها | مقایسه linear / rbf / poly / sigmoid |
| 📊 ارزیابی مدل | Confusion Matrix، Classification Report، تأثیر C |

## 🚀 اجرا روی لوکال

```bash
git clone https://github.com/YOUR_USERNAME/svm-explorer
cd svm-explorer
pip install -r requirements.txt
streamlit run app.py
```

## 🤗 Hugging Face Spaces

این اپ روی Hugging Face Spaces هم قابل دسترس است:
👉 [Open in Spaces](https://huggingface.co/spaces/YOUR_USERNAME/svm-explorer)

## 📦 تکنولوژی‌ها

- **Streamlit** — رابط کاربری
- **scikit-learn** — مدل SVM
- **Matplotlib / Seaborn** — نمودارها
- **NumPy / Pandas** — پردازش داده

## 🎛️ پارامترهای قابل تنظیم

- **C**: پارامتر Regularization (0.01 تا 20)
- **Kernel**: linear, rbf, poly, sigmoid
- **Gamma**: scale / auto / custom
- **Degree**: برای kernel چندجمله‌ای
- **Dataset**: 5 نوع dataset مصنوعی و واقعی
- **Test Size**: نسبت تقسیم داده

## 📁 ساختار پروژه

```
svm-explorer/
├── app.py              # اپ اصلی Streamlit
├── requirements.txt    # کتابخانه‌های مورد نیاز
└── README.md           # این فایل
```

---
Made with ❤️ using Streamlit & scikit-learn
