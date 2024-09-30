

class time:
    def __init__(self, h=0, m=0):
        self._hour = h
        self._minute = m

    def set_time(self, h, m):
        self._hour = h
        self._minute = m

    def __str__(self):
        print(str(self._hour) + ":" + str(self._minute) )


class Date:
    def __init__(self, d, m, y):
        self._day = d
        self._month = m
        self._year = y

    def __str__(self):
        print(str(self._day) + "/" + str(self._month) + "/" + str(self._year))

    def __eq__(self, other):
        return self._day == other._day and self._month == other._month and self._year == other._year

class Order:
    _order_num = 1
    
    def __init__(self, d, m, y, h, minute, cost=50):
        self._t = time(h, minute)
        self._d = Date(d, m, y)
        self._order_id = _order_num
        _order_num += 1
        self._cost = cost

    def __gt__(self, other):
        return self._cost > other._cost

    def getTime(self):
        return self._time

    def getDate(self):
        return self._date

class CashRegister():
    def __init__(self):
        self._orders = []
        
    def monthly_total_income(self, month):
        total = 0
        for order in self._orders:
            if order._date._month == month:
                total += order._cost
        return total
    
    def most_expensive_order(self, date):
        max_order_amount = 0
        max_order_number = 0
        for order in self._orders:
            if order._date == date: # will use the __eq__ method
                if order._cost > max_order_amount:
                    max_order_amount = order._cost
                    max_order_number = order._order_id
        return max_order_number
    def less_than(self, cost):
        order_list = []
        for order in self._orders:
            if order._cost < cost: 
                order_list.append(order)
        if len(order_list)==0: return None
        return order_list
        
        
    

    def addItem(self, price):
        self._itemCount = self._itemCount + 1
        self._totalPrice = self._totalPrice + price

    def getTotal(self):
        return self._totalPrice

    def getCount(self):
        return self._itemCount

    def clear(self):
        self._itemCount = 0
        self._totalPrice = 0.0