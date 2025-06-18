import json

plate = input("Enter plate number: ").upper().replace(" ", "")
owner = input("Enter owner name: ")

try:
    with open("database.json", "r") as f:
        db = json.load(f)
except FileNotFoundError:
    db = {}

db[plate] = owner

with open("database.json", "w") as f:
    json.dump(db, f, indent=4)

print(f"✅ Registered {plate} -> {owner}")
