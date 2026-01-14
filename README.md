# 🛡️ مَرْتَكَز - MortakizAi
**The Power of Local AI, Centered in Your Machine.**

[![Python Version](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Powered](https://img.shields.io/badge/GPU-NVIDIA%20H100%20%7C%20A4000-green.svg)](https://www.nvidia.com/)

**مَرْتَكَز (MortakizAi)** هو محرك ذكاء اصطناعي محلي فائق السرعة، صُمم ليكون "نقطة الارتكاز" المستقرة لعملية التطوير البرمجي داخل بيئات VSCode عبر إضافات مثل **Cline** و **Continue**. يوفر المشروع توافقاً كاملاً مع OpenAI API مع ميزة التكيف الذكي مع العتاد المتوفر.

---

## ✨ المميزات الرئيسية (Key Features)

- 🔒 **خصوصية مطلقة (Local-First):** بياناتك وأكوادك لا تخرج من جهازك أبداً.
- 🚀 **ذكاء تكيّفي (Adaptive Performance):** نظام ذكي يكتشف قوة كرت الشاشة ويفعل نمط التشغيل الأمثل:
  - **High-Power Mode (BF16):** مخصص لكروت H100 و A100 لأداء خارق.
  - **Efficient Mode (4-bit):** مخصص لكروت A4000 و RTX 4060 لسرعة عالية مع استهلاك أقل للذاكرة.
- 🛠️ **منع التكرار (Anti-Looping):** معالجة متقدمة لرموز التوقف (Stop Tokens) لمنع حلقات الاعتذار والتكرار في Cline.
- ⚡ **بث لحظي (Instant Streaming):** دعم كامل للـ Streaming لظهور الكود فور توليده.

---

## 🏗️ المعمارية (Architecture)

يعمل "مرتكز" كجسر يربط بين أقوى موديلات البرمجة (`Qwen2.5-Coder`) وبين أدواتك المفضلة:
`VSCode (Cline) <---> MortakizAi API <---> Local GPU Acceleration`

---

## 🚀 البدء السريع (Quick Start)

### 1. المتطلبات
تأكد من وجود بيئة Python 3.12.3 وتثبيت المكتبات اللازمة:
```bash
pip install torch transformers fastapi uvicorn bitsandbytes accelerate flash-attn
