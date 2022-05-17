# Spring-2022
Master's capstone project: Determining a hedge fund's strategy and performance based on given transactional data. 

What you are given:  
1. Stocks file   
•PERMNO = security identifier used by CRSP  
•RET2 = Daily gain of fund’s position in the security  
•SIC code from CRSP  
•MCAP = market value of security at end of day (in thousands, price times shares
outstanding)  
•Net shares = position in the security  

2. Portfolio file  
•RET2 = daily gain of fund’s positions assuming 100% collateral posted for short positions  
•RET3 = daily gain of fund’s positions assuming 0% collateral posted for short positions  
•RET2-long = daily gone of long side only (rebalanced)  
•RET2-short = daily gone of short side only (rebalanced)  
•VALUE-Close = sum of absolute values of security positions at the end of the day (as
in RET2)  
•VALUE-Long = sum of values of long positions at the end of the day.  
VALUE-Short = sum of values of short positions at the end of the day.  
The net of these long and short values can be used as a base for RET3 weights.  

Identify the securities using CRSP:  
•Use PERMNO in CRSP  
– Stocks of U.S. firms, REITs, or ETFs  
– Can access financial statements (link to Compustat)  
– Can access sell-side earnings forecasts (link to I/B/E/S)  

Get acquainted:  
•How many securities are held on average?  
•Examine distribution of SIC codes.  
•How much time are positions typically held?  
•How many funds are held versus specific stocks? (ETFs have SIC=6276)  
•How well does the fund perform (against benchmarks)?  
•Next: Calculate daily weights for each security (Collect price data from CRSP. Use absolute
value of “prc”).  

Perspective:  
•The analysis has several layers. Do not underestimate the workload.  
1. Learn the portfolio composition.  
2. Assess the fund’s performance. [If poor, then end.]  
3. If performance is good, determine whether the fund’s performance is replicable. [If
replicable, then end.]  
4. If not replicable, characterize how the fund generates alpha. What does the fund do
well? Do you believe the fund displays investing skill (not luck)?  
5. Do you see any areas where the fund can improve?  
