@echo off
chcp 65001 >nul
REM Doğal Dil İşleme - Chunking projesi - tek tıkla çalıştırma (Windows)
echo ============================================
echo  Gerekli paketler kuruluyor...
echo ============================================
python -m pip install -r requirements.txt
echo.
echo ============================================
echo  Boru hatti calistiriliyor (indir-egit-test)...
echo ============================================
python src\run_all.py
echo.
echo Bitti. Ciktilar: data\chunks, models, results
pause
