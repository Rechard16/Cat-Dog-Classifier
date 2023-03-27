@echo off
set son=%cd%
@Rem 获取上级目录，暂时没使用到
pushd %son%
cd ..
set parent=%cd%
popd

for /d %%i in (*) do (
move /y %son%\%%i\*.* %son%
rd %son%\%%i
)
@Rem pause