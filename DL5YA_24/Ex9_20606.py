
def go_up(lst, i):
    while i<len(lst)-1 and lst[i] < lst[i+1]: i+=1   
    return i

def go_down(lst, i):
    while i<len(lst)-1 and lst[i] > lst[i+1]: i+=1   
    return i


def max_drop(lst):
    i, j, hi_index, max_drop = 0, 0, 0, 0
    while j <len(lst)-1:
        i = go_up(lst, j+1)
        j = go_down(lst, i+1)
        max_drop = max(max_drop, lst[i] - lst[j], lst[hi_index] - lst[j])
        if lst[i]>lst[hi_index]: hi_index = i
    return max_drop

lst = [5, 21, 3, 27, 12, 24, 7, 6, 4] 
#lst = [5, 21, 3, 22, 12, 7, 26, 14] 
#lst = [5, 15, 3, 22, 12, 7, 27, 14] 
print (max_drop(lst))