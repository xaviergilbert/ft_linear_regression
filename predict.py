
def main():
    f = open("theta.txt", "r")
    line = f.readlines()
    b, a = line[0].rstrip(), line[1].rstrip()
    while True:

        user_input = input("What is the mileage of your car ?\n")
        if user_input.isnumeric():
            break
        else:
            print("Enter a number please.")
    print("Your car is probably worth", int(float(user_input) * float(a) + float(b)), "euros")

if __name__ == "__main__":
    main()