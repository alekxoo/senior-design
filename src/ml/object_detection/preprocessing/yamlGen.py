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
