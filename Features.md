#FEATURE LIST FINAL

* Numerical Store Id: Which store is it? Some stores might always have higher sales
* Categorical Store type: Store size etc
* Categorical Assortment Level: Does it contain a lot of items of each type? Assortment of items.
* Numerical Competiion Distance (measure of distance): The farther away the competition, the better
* Numerical Competition Open Since Month 11: New competitors are less risky. -> Convert into relative months
* Numerical Competition Open Since Year (more is worse): -> Convert into relative months
* Categorical if promo on that day: 
* Categorical if promo is consecutive: Has promo been going on
* Numerical promo2sinceyear: How much time has promo been going on? -> Convert into relative months
* numerical promo2sinceweek: How much time has promo been going on? -> Convert into relative months
* Categorical promointerval p_jan, p_feb, p_mar, p_april: 12 bit vector of months  Don't know how to make sense of this?
* numerical month_today
* numerical year_today
* numerical day_today
* numerical day of week (can’t be categorical since sequences etc)
* categorical if_school_ holiday: School holiday should increase the sales, right?
