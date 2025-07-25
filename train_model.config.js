module.exports = {
  apps: [
    {
      name: 'train-model',
      script: 'train_model.py',
      args: 'all --csv_file sample_bitcoin_data1.csv',
      interpreter: 'python3',
      watch: false,
      autorestart: true,
      max_restarts: 3,
      env: {
        // Add any environment variables here if needed
      }
    }
  ]
};
