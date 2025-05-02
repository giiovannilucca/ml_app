@echo off
echo Criando ambiente virtual...
python -m venv venv

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo Instalando dependências do requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Ambiente pronto! Para ativar depois, use:
echo     venv\Scripts\activate.bat
pause