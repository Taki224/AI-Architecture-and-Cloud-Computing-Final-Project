"""
Temporary script to generate training data for Isolation Forest models
Generates large dataset using same sensor simulation logic as GUI
"""
import csv
import time
from sensor_simulator import VibrationSensor


def generate_training_data(
    num_samples: int = 100000,
    output_file: str = "training_data.csv"
):
    """
    Generate training data and save to CSV file.
    
    Args:
        num_samples: Number of samples to generate (default: 100,000)
        output_file: Output CSV filename
    """
    print("=" * 60)
    print("Training Data Generation for Isolation Forest Models")
    print("=" * 60)
    print(f"\nGenerating {num_samples:,} samples...")
    print(f"Output file: {output_file}\n")
    
    # Initialize sensor with higher anomaly rate for better training
    # Using 15% anomaly rate to give model enough examples to learn from
    # In real deployment, actual anomaly rate will be much lower (~0.3%)
    sensor = VibrationSensor(
        mean=0.0,
        std_dev=1.0,
        anomaly_rate=0.15,  # 15% for training (much higher than production)
        small_anomaly_range=(3.0, 4.0),
        large_anomaly_range=(5.0, 8.0),
        large_anomaly_ratio=0.4
    )
    
    # Open CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['timestamp', 'value', 'is_anomaly', 'severity'])
        
        # Generate data
        start_time = time.time()
        for i in range(num_samples):
            # Generate reading
            value, is_anomaly, severity = sensor.generate_reading()
            timestamp = time.time()
            
            # Write to CSV
            writer.writerow([timestamp, value, int(is_anomaly), severity])
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (num_samples - i - 1) / rate
                print(f"Progress: {i+1:,}/{num_samples:,} samples "
                      f"({(i+1)/num_samples*100:.1f}%) - "
                      f"Rate: {rate:.0f} samples/sec - "
                      f"ETA: {remaining:.1f}s")
    
    # Print statistics
    elapsed_total = time.time() - start_time
    stats = sensor.get_statistics()
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nTotal samples: {stats['total_readings']:,}")
    print(f"Total anomalies: {stats['total_anomalies']:,} ({stats['anomaly_rate']:.2f}%)")
    print(f"  - Small anomalies: {stats['small_anomalies']:,}")
    print(f"  - Large anomalies: {stats['large_anomalies']:,}")
    print(f"\nTime elapsed: {elapsed_total:.2f} seconds")
    print(f"Generation rate: {num_samples/elapsed_total:.0f} samples/sec")
    print(f"\nData saved to: {output_file}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train heavy model (Isolation Forest n=200)")
    print("2. Train light model (Isolation Forest n=10)")
    print("3. Save models as model_heavy.pkl and model_light.pkl")
    print("=" * 60)


if __name__ == "__main__":
    # Generate 100,000 samples with balanced anomaly distribution
    # This gives ~15,000 anomalies for training (15% rate)
    # Note: Production anomaly rate is much lower (~0.3%), but training
    # needs more examples to learn anomaly patterns effectively
    generate_training_data(
        num_samples=100000,
        output_file="training_data.csv"
    )
    
    # Generate validation set with same balanced distribution
    print("\nGenerating validation dataset...")
    generate_training_data(
        num_samples=20000,
        output_file="validation_data.csv"
    )
