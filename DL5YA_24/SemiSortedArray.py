
# Skip zeros to the left (if possible) or right (if not) of the current element
def skipZeros(a, i, up=True):
    if up:
        while (a[i] == 0 and i < len(a)): i += 1
    else:  
        while (a[i] == 0 and i >= 0): i -= 1
    return i

def FindSemiSorted(a, b, e, num):
    # Skip leading and trailing zeros
    b = skipZeros(a, b, True)
    e = skipZeros(a, e, False)
    while (b<=e):
        mid = (b + e) // 2

        mid = skipZeros(a, mid, True)
        if mid == len(a): mid = skipZeros(a, mid-1, False)
 
        if a[mid] == num:
            return mid
     
        
        if a[mid] > num:
            e = skipZeros(a, mid - 1, False)
        else:
            b = skipZeros(a, mid + 1, True)
    return -1


l = [3,0,0,4,7,9,0,0,0,0,11,15,0,19,20,0,0,31,40,0,50]
print (FindSemiSorted(l, 0, len(l)-1, 3))