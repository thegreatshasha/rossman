#FEATURE LIST FINAL

* Categorical Store Id: Which store is it? Some stores might always have higher sales
* Categorical Store type: Store size etc
* Categorical Assortment Level: Does it contain a lot of items of each stype?
* Numerical Competiion Distance (measure of distance): Larger the better
* Numerical Competition Open Since Month 11: The worse the b: Competition has been open for how much time?
* Numerical Competition Open Since Year (more is worse): -> These should be converted into how many months the competition has been open
* Categorical if promo on that day: 
* Categorical if promo is consecutive
* Numerical promo2sinceyear: How many months has been promo going time?
* numerical promo2sinceweek: How many months has been promo going on
* p_jan, p_feb, p_mar etc to be encoded as vector of  (is the offer going on in march): 12 of these basically 22 features
* numerical month_today
* numerical year_today
* numerical day_today
* numerical day of week (can’t be categorical since sequences etc)
* categorical if_school_ holiday: School holiday should increase the sales, right?
* Is current day a promo month: Current date a promo month?
* Is current month a promo month?
* Is current year a promo year?