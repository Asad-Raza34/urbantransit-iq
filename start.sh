# UrbanTransit IQ - Startup Script
# Run this to set up the environment and launch the dashboard

# Activate conda environment
conda activate urbantransit-iq

# Set environment variables
export UTIQ_HADOOP_MODE=local
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export UTIQ_HADOOP_HOME=/opt/hadoop

# Launch dashboard
streamlit run app/main.py