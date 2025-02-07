import yaml
import uuid

def get_user_input():
    full_name = input("Enter your full name: ")
    num_cars = int(input("Enter number of cars: "))
    
    classes = []
    for i in range(num_cars):
        label = input(f"Enter label for car {i+1}: ")
        classes.append({"id": i+1, "label": label})
    
    data = {
        "user": {"full_name": full_name},
        "num_classes": num_cars,
        "classes": classes
    }
    
    filename = f"config_{uuid.uuid4().hex[:8]}.yaml"
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"Configuration saved as {filename}")

def main():
    get_user_input()

if __name__ == "__main__":
    main()


'''
Process for front end:

assuming that the user can upload 1 car at a time

1. User logs into account(we get user's name and unique ID(if needed))
2. User creates new race (This will be a new folder on their profile)
3. Create a new Folder (E.g. race-xxxx, with subfolders (e.g images, weights, config))
3. User uploads batch of photos for Car 1 (these photos would be stored in that race folder under the images subfolder; we will data split later in training)
4. User inputs label for the car (that is name of the folder of images)
5. User submits above information for Car 1
6. yamlGen.py parses submitted information to get num_classes + 1 and add id+ class label for Car 1

(Repeat steps 2-4 for as many cars he wants to upload until finished)

6. yaml file = race-xxxx -> config 
7. For each class = race-xxxx -> images -> [class_label_created_by_user]
8. Start training script (should generate train/validation subfolders within race-xxxx -> images)
9. Generated model weights after training = race-xxxx -> weights
'''