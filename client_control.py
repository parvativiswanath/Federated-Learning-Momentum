import torch
import csv
import re
import math
import client
import serial
import alert
from datetime import datetime
import time
from collections import deque
import subprocess

def save_to_csv(data, prediction=None):
    with open('sensor_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #row = [timestamp] + data
        row = data
        if prediction is not None:
            row.append(prediction)
        writer.writerow(row)

if __name__ == "__main__":
   
    model = client.SimpleNN()  # Initialize the model
    print('\nIntializing model',model)
    # torch.save(model.state_dict(), "global_params.pth")
    # velocity = {name: torch.zeros_like(param) for name,param in model.named_parameters()}
   
    # print('\nPRE-TRAINING\n')
    # for i in range(5):
    #     client.train_model()
    #     client.send_params()

    #subprocess.run(["python", "client.py", "paddy_field_client2.csv"], check=True)

    # Buffer to store last 5 readings
    data_buffer = deque(maxlen=10)

    while(True):

        print('\nLOAD TRAINED MODEL\n')
        model.load_state_dict(torch.load('received_models/updated_model_params.pth'))
        model.eval()  # Set model to evaluation mode

        print('\nCOLLECT DATA AND INFERENCE\n')
        # Configure serial connection
        # ser = serial.Serial(
        #     port='/dev/ttyUSB0',
        #     baudrate=9600,
        #     timeout=1
        # )

        try:
            # Write header to CSV file
            with open('sensor_data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Water Level', 'Moisture Percentage', 
                            'Light Intensity', 'Temperature', 'Humidity', 'Cluster'])
            
            print("Starting data collection...")
            data_buffer.clear()
            while True:
                #if ser.in_waiting > 0:
                    # Read and parse data
                    #line = ser.readline().decode('utf-8').strip()
                    line = "Water Level:100, Moisture Percentage:37, Light Intensity: 876, Temperature:28, Humidity: 82"
                    print('Received data: ',line)
                    
                    if 'nan' in line.lower():
                        print("NaN detected in raw data, discarding row.")
                        continue  # Skip this row
                    
                    # Extract only numerical values using regex
                    data = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # Matches both integers and floats

                    # Convert extracted values to floats
                    float_data = [float(x) for x in data]
                    
                    print('Sensor data values:',float_data)
                    
                    # Check if any value is NaN
                    if any(math.isnan(x) for x in float_data):
                        print("NaN detected, discarding row.")
                        continue  # Skip this row and wait for the next one
                    
                    data_buffer.append(float_data)
                    
                    # Make prediction if we have enough data
                    prediction = None
                    
                    if len(data_buffer) == 5:  # Wait for buffer to fill
                        # Prepare input tensor
                        features = torch.FloatTensor(float_data)
                        
                        # Make prediction
                        with torch.no_grad():
                            prediction = model(features)
                            prediction = prediction.item()
                        
                        break
                    
                    # Save data and prediction to CSV
                    save_to_csv(data, prediction)
                    
        except KeyboardInterrupt:
            print("\nData collection stopped by user")
            #ser.close()
        except Exception as e:
            print(f"An error occurred: {e}")
            #ser.close()

        if prediction is not None:
            # Print status
                status = f"Data received: {line}"
                if prediction is not None:
                    try: {print('\n\n####################################\nPREDICTION\n####################################\n\n CLuster 0: ',prediction[0], ', Cluster 1: ',prediction[1], ', Cluster 2: ',prediction[2])}
                    except: {print('Error when printing prediction')}

                    if prediction[0] > 0:
                       status += f"\nPredicted irrigation need: No irrigation required\n"
                    if prediction[1] > 0:
                        status += f"\nPredicted irrigation need: Moderate irrigation recommended\n"
                    if prediction[2] > 0:
                        status += f"\nPredicted irrigation need: Immediate irrigation recommended!\n"
                        alert.send_alert()
                        exit()
                print(status + "\n")

        subprocess.run(["python", "client.py", "sensor_data.csv"], check=True)



