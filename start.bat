@echo off
REM UrbanTransit IQ - Windows Startup Script

REM Activate conda environment
call conda activate urbantransit-iq

REM Set environment variables
set UTIQ_HADOOP_MODE=local
set JAVA_HOME=C:\Java\jdk-17.0.20.1+1
set UTIQ_HADOOP_HOME=%CD%\hadoop

REM Launch dashboard
streamlit run app/main.py