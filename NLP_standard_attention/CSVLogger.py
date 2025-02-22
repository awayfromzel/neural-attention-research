import csv
import os

class CSVLogger:
    def __init__(self, folder="Logs", filename="training_metrics.csv"):
        # Ensure the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Define the full path to the CSV file
        self.csv_file = os.path.join(folder, filename)
        
        # Initialize the CSV file and write the header
        self.init_csv()

    def init_csv(self):
        # Create the file if it doesn't exist, or append if it does
        file_exists = os.path.isfile(self.csv_file)
        
        # If the file does not exist, we write the header
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                # Write the CSV header
                writer.writerow([
                    "Date & Time", "Iteration", "Running Time (s)", 
                    "Training Loss", "Validation Loss", "Perplexity", "Learning Rate", 
                    "BPC", "Throughput", "Gradient Norm", 
                    "Inference Time (s)", "Inference Time per Sample (s)", 
                    "GPU Memory Allocated (MB)", "GPU Memory Reserved (MB)"
                ])

    def log(self, current_time, iteration, running_time, training_loss, val_loss, 
            perplexity, learning_rate, bpc, throughput, gradient_norm, 
            inference_time, inference_time_per_sample, memory_allocated, memory_reserved):
        # Write the data to the CSV file
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                current_time, iteration, running_time, training_loss, val_loss, perplexity, 
                learning_rate, bpc, throughput, gradient_norm, 
                inference_time, inference_time_per_sample, 
                memory_allocated, memory_reserved
            ])