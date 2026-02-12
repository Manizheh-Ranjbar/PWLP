## PWLP: Scalable Edge-Centric Subgraph Learning via Pool Walks for Link Prediction



pip freeze > requirements.txt



۲. ساختن خودکار محیط با یک فایل اسکریپت
یک فایل بساز که:
پوشهٔ venv را بسازد (اگر نباشد)،
آن را فعال کند و
از روی requirements.txt نصب کند.
مثال برای ویندوز (PowerShell) – مثلاً اسم فایل: setup.ps1 یا setup_env.bat:
# ساخت venv اگر نباشد
if (!(Test-Path "venv")) {
    python -m venv venv
}
# فعال‌سازی و نصب
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# ساخت venv اگر نباشدif (!(Test-Path "venv")) {    python -m venv venv}# فعال‌سازی و نصب.\venv\Scripts\Activate.ps1pip install -r requirements.txt
یا به‌صورت Batch – مثلاً setup.bat:
if not exist venv python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist venv python -m venv venvcall venv\Scripts\activate.batpip install -r requirements.txt
بعداً کافی است این فایل را اجرا کنی تا همان environment ساخته شود.
