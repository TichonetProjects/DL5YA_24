
def is_in(digit, number):
    for i in number:
        if i == digit: return True
    return False

def is_in_rec(digit, number, i):
    if i>=len(number): return False
    return number[i] == digit or is_in_rec(digit, number, i+1)

def bulls_and_cows(number, guess):
    bulls = cows = 0
    for i in range(len(number)):
        if guess[i] == number[i]:
            bulls += 1
        elif is_in_rec(guess[i], number, 0):
            cows += 1
    return bulls*2 + cows

guess = [3,1,5,4]        
number = [7,2,5,1]

print(bulls_and_cows(number, guess)) # (0, 1)