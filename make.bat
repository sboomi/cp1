@ECHO OFF

@REM Makescript for Windows

SET PYTHON_INTERPRETER=python
SET PROCESSED=.\data\processed\
SET RAW=.\data\raw\
SET RAW_CSV=%RAW%comments_train.csv
SET CLEAN_CSV=%PROCESSED%comments_clean.csv
SET MODEL_FOLDER=.\models\
SET OLD_MODEL=%MODEL_FOLDER%sentiment_pipe.joblib
SET EPOCHS=50
SET TRAIN_BS=256
SET VAL_BS=64
SET SEED=32451365

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  data       to make a clean dataset
	echo.  train      to train a ML model
	echo.  traindeep  to train a NN model
	goto end
)

if "%1" == "data" (
    %PYTHON_INTERPRETER% .\src\data\make_dataset.py %RAW% %PROCESSED% %RAW_CSV%
    if errorlevel 1 exit /b 1
    echo.
	echo.Build finished; clean dataset created.
	goto end
)

if "%1" == "train" (
    %PYTHON_INTERPRETER% .\src\train_ml_model.py ^
	%CLEAN_CSV% ^
	%MODEL_FOLDER% ^
	-cm %OLD_MODEL%
	if errorlevel 1 exit /b 1
    echo.
	echo.Build finished; ML models created.
    goto end
)

if "%1" == "traindeep" (
    %PYTHON_INTERPRETER% .\src\train_nn_model.py ^
	%CLEAN_CSV% ^
	%MODEL_FOLDER% ^
	-cm %OLD_MODEL% ^
	--batch-size %TRAIN_BS% %VAL_BS% ^
	--epochs=%EPOCHS%

	if errorlevel 1 exit /b 1
    echo.
	echo.Build finished; ML models created.
    goto end
)

:end