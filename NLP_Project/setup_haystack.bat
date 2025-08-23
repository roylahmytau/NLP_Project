@echo off
echo Creating virtual environment...
python -m venv nlp_venv

echo Activating virtual environment...
call nlp_venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages from requirements.txt...
pip install -r requirements.txt

echo Setup complete!
echo To activate your environment in the future, run: nlp_venv\Scripts\activate.bat
pause