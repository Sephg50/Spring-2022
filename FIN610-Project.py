# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:07:40 2022

@author: seph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import wrds
import statistics
import time

start_time = time.time() # <-- imported time to calculate speed efficiency
print('Timing:')

### 1. Individual positions dataframe ###

"""
PERMNO = security IDentifier used by CRSP
mcap = market value of firm's equity at the end of day
sic = SIC code from CRSP
RET2 = Daily gain of fund's position in the security
net_shares = position in the security
"""

positions_df = pd.read_csv('Stocks_2021.csv',
                      index_col = 'DATE',
                      parse_dates = True).drop(['date_txn'], 1)

positions_df['ticker'] = positions_df['ticker'].ffill() # <-- Fill in tickers
positions_df = positions_df.loc['2010-05-03':] # <-- Begin data at May 5, 2010 per project guidelines
positions_df = positions_df.sort_index() # <-- Sort df by date

positions_df = positions_df[(positions_df.ret2 != 'R') & (positions_df.ret2 != '.')] # <-- removing erraneous rows from dataframe
positions_df['ret2'] = positions_df['ret2'].astype(float) # <-- correcting ret2 column to float values

### SQL to import prc (end of day prices) and calculate position market values

conn = wrds.Connection(wrds_username='seph')
begdate = '05/03/2010'
enddate = '01/31/2014'

# _crsp = conn.raw_sql("""
#                       SELECT date, permno, PRC
#                       from crsp.dsf
#                       WHERE permno in %(permnos)s
#                       and date between %(beg)s and %(end)s
#                       """, date_cols = ['date']
#                       , params = {'beg':begdate, 'end':enddate, 'permnos':tuple(positions_df['PERMNO'].to_list())})

# _crsphdr = conn.raw_sql("""
#                       SELECT permno, HCOMNAM
#                       from crsp.dsfhdr
#                       WHERE permno in %(permnos)s
#                       """, params = {'permnos':tuple(positions_df['PERMNO'].to_list())})


# find = conn.describe_table(library='crsp', table='holdings_co_info')

# find2 = conn.list_tables(library = 'crsp')

# iwm = conn.raw_sql("""
#                     SELECT permno, security_name
#                     from crsp.holdings_co_info
#                     WHERE permno in %(permnos)s
#                     """, params = {'permnos':tuple(positions_df['PERMNO'].to_list())})
                    
# print(iwm)

# _crsp = _crsp.set_index('date')
# _crsp.index.names = ['DATE']
# _crsp = _crsp.rename(columns = {'permno': 'PERMNO'})
# _crsphdr = _crsphdr.rename(columns = {'permno': 'PERMNO', 'hcomnam' : 'ID'})

# _crsp.to_csv('crsp.csv')
# _crsphdr.to_csv('crsphdr.csv') # <-- export and import database as csv to improve speed (database is unchanged, only need to import the first time)

crsp = pd.read_csv('crsp.csv',
                   index_col = 'DATE',
                   parse_dates = True)
crsphdr = pd.read_csv('crsphdr.csv')
crsphdr = crsphdr.iloc[:, 1:] # <-- remove extraneous first column created from export/import csv

key_df = crsphdr.set_index('PERMNO')
key_df = key_df.merge(
    positions_df[['PERMNO', 'ticker']].drop_duplicates(
        subset=['PERMNO']), how = 'left', on = ['PERMNO']).set_index('ticker') # <-- Create a helper Key to quickly identify positions by ticker, PERMNO, ID

"""
PERMNO was used to pull correct company data from database; some repeated tickers therefore PERMNO still needed:
    VVI
        19721 = VIAD CORP
        20117 = LIVERAMP HOLDINGS INC
    CVU
        77900 = CPI AEROSTRUCTURES INC
        78044 = UNIFI INC
    QTM
        87043 = QUANTUM CORP
        87165 = MOVE INC
    IWO
        88401 = ISHARES TRUST
        88417 = AXCELIS TECHNOLOGIES INC
    TDS
        63773 = TELEPHONE AND DATA SYSTEMS INC
        90655 = TELEPHONE AND DATA SYSTEMS INC ???
    BWY
        92109 = BWAY HOLDING CO
        92184 = PHARMERICA CORP
"""

key_df.loc['IWM','ID'] = 'ISHARES RUSS 2000 ETF'
key_df.loc[(key_df.index == 'IWO') & (key_df['PERMNO'] == 88401), 'ID'] = 'ISHARES RUSS 2000 GROWTH ETF'
key_df.loc['IGE', 'ID'] = 'ISHARES NA NATURAL RESOURCES ETF'
key_df.loc['IYR', 'ID'] = 'ISHARES US REAL ESTATE ETF'
key_df.loc['IJR', 'ID'] = 'ISHARES CORE SNP SMALLCAP ETF'
key_df.loc['IWN', 'ID'] = 'ISHARES RUSS 2000 VALUE ETF'
key_df.loc['TUR', 'ID'] = 'ISHARES MSCI TURKEY ETF'
key_df.loc['EWG', 'ID'] = 'ISHARES MSCI GERMANY ETF'
key_df.to_csv('key.csv') 

positions_df = crsp.merge(positions_df, on = ['DATE', 'PERMNO'])
positions_df = positions_df.reset_index().merge(crsphdr, how = 'left', on = ['PERMNO']).set_index('DATE') # <-- merge hcomnam data while maintaining datetimeindex

positions_df = positions_df.drop(['ID'], 1).reset_index().merge(key_df.reset_index()[['PERMNO','ID']], how = 'left', on = ['PERMNO']).set_index('DATE') # <-- use updated IDs from key_df in positions_df (for ETFs) 

positions_df['position_value'] = positions_df['prc'] * positions_df['net_shares']

### Analysis of how many positions are held on average

dailynumberpositions_df = positions_df['PERMNO'].resample('D').nunique().to_frame() # <-- number of unique PERMNOs each day
dailynumberpositions_df = dailynumberpositions_df.replace(0,np.nan).ffill()
dailynumberpositions_df.to_csv('FIN610-DailyNumberOfPositions.csv')
dailynumberpositions_df.plot(figsize = (18,6)).set_title('Daily number of unique positions from May 2010 to January 2014')
plt.savefig('FIN610-Daily number of unique positions from May 2010 to January 2014.png')
print(f'Average daily number of unique PERMNOs = {dailynumberpositions_df["PERMNO"].mean()}')

### Analysis of portfolio weights

sic_df = positions_df.pivot_table(index = 'DATE', columns = 'sic', values = 'position_value', aggfunc='sum')
weights_SIC_df = sic_df.div(sic_df.sum(axis=1), axis=0) # <-- change values to percentage
weights_SIC_df.to_csv('FIN610-weightsSIC.csv')

ID_df = positions_df.pivot_table(index = 'DATE', columns = 'ID', values = 'position_value', aggfunc='sum')
# ID_df.to_csv('FIN610-individual position market values.csv')
weights_ID_df= ID_df.div(ID_df.sum(axis=1), axis=0) 
weights_ID_df.to_csv('FIN610-weightsID.csv')

PERMNO_df = positions_df.pivot_table(index = 'DATE', columns = 'PERMNO', values = 'position_value', aggfunc='sum')
weights_PERMNO_df = PERMNO_df.div(PERMNO_df.sum(axis=1), axis=0) 
weights_PERMNO_df.to_csv('FIN610-weightsPERMNO.csv')


rets_ID_df = positions_df.pivot_table(index = 'DATE', columns = 'ID', values = 'ret2')
weightedrets_ID_df = rets_ID_df * weights_ID_df.shift(1) # today's returns with yesterday's weights
weightedrets_ID_df.to_csv('FIN610-weightedrets_ID.csv')


weighted_cumrets_ID_df = pd.DataFrame().reindex_like(weightedrets_ID_df)
weighted_cumrets_ID_df = (1 + weightedrets_ID_df).cumprod() - 1
weighted_cumrets_ID_df.to_csv('FIN610-weightedcumrets_ID.csv')

finalcumrets_ID_df = weighted_cumrets_ID_df.ffill(axis=0).iloc[-1,:]
finalcumrets_ID_df.to_csv('FIN610-FinalCumRets_ID.csv')

rets_PERMNO_df = positions_df.pivot_table(index = 'DATE', columns = 'PERMNO', values = 'ret2')
weightedrets_PERMNO_df = rets_PERMNO_df * weights_PERMNO_df.shift(1)

pivot_weightedrets_PERMNO_df = weightedrets_PERMNO_df.reset_index(
    ).melt(id_vars = ['DATE'], value_vars = weightedrets_PERMNO_df.columns, value_name = 'wRet2').set_index('DATE')
"""Created helper df to merge wRet2 into positions_df"""

positions_df = positions_df.merge(pivot_weightedrets_PERMNO_df, how = 'inner', on = ['DATE','PERMNO'])

sic_wrets_df = positions_df.pivot_table(index = 'DATE', columns = 'sic', values = 'wRet2', aggfunc='sum')
industries_df = pd.DataFrame(index=sic_df.index)
industries_wrets_df = pd.DataFrame(index=sic_wrets_df.index)
## Industries categorized by Fama French 48 Industrial SIC Codes

agric_xs = [] #1. Agriculture
food_xs = [] #2. Food Products
soda_xs = [] #3. Candy and Soda
beer_xs = [] #4. Beer and Liquor
smoke_xs = [] #5. Tobacco Products
toys_xs = [] #6. Recreation
fun_xs = [] #7. Entertainment
books_xs = [] #8. Printing and Publishing
hshld_xs = [] #9. Consumer Goods
clths_xs = [] #10. Apparel
hlth_xs = [] #11. Healthcare
medeq_xs = [] #12. Medical Equipment
drugs_xs = [] #13. Pharmaceutical Products
chems_xs = [] #14. Chemicals
rubbr_xs = [] #15. Rubber and Plastic Products
txtls_xs = [] #16. Textiles
bldmt_xs = [] #17. Construction Materials
cnstr_xs = [] #18. Construction
steel_xs = [] #19. Steel Works Etc
fabpr_xs = [] #20. Fabricated Products
mach_xs = [] #21. Machinery
elceq_xs = [] #22. Electrical Equipment
autos_xs = [] #23. Automobiles and Trucks
aero_xs = [] #24. Aircraft
ships_xs = [] #25. Shipbuilding, Railroad Equipment
guns_xs = [] #26. Defense
gold_xs = [] #27. Precious Metals
mines_xs = [] #28. Non-Metallic and Industrial Metal Mining
coal_xs = [] #29. Coal
oil_xs = [] #30. Petroleum and Natural Gas
util_xs = [] #31. Utilities
telcm_xs = [] #32. Communication
persv_xs = [] #33. Personal Services
bussv_xs = [] #34. Business Services
comps_xs = [] #35. Computers
chips_xs = [] #36. Electronic Equipment
labeq_xs = [] #37. Measuring and Control Equipment
paper_xs = [] #38. Business Supplies
boxes_xs = [] #39. Shipping Containers
trans_xs = [] #40. Transportation
whlsl_xs = [] #41. Wholesale
rtail_xs = [] #42. Retail
meals_xs = [] #43. Restaurants, Hotels, Motels
banks_xs = [] #44. Banking
insur_xs = [] #45. Insurance
rlest_xs = [] #46. Real Estate
fin_xs = [] #47. Trading
other_xs = [] #48. Sanitary Services <-- Classified as "Almost Nothing" but SIC 4950 Sanitary Services is the only SIC present from this category
nce_xs = [] #49. SIC 9999 represented 

for sic in sic_df:
    if (100 <= sic <= 299) or (700 <= sic <= 799) or (910 <= sic <= 919) or (sic == 2048): #1 agric
        agric_xs.append(sic)
        industries_df['Agriculture'] = sic_df[agric_xs].sum(axis=1)
        industries_wrets_df['Agriculture'] = sic_wrets_df[agric_xs].sum(axis=1)
    elif ((2000 <= sic <= 2046) or (2050 <= sic <= 2063) or (2070 <= sic <= 2079) or (2090 <= sic <= 2092) or (sic == 2095) 
        or (2098 <= sic <= 2099)): #2 food
        food_xs.append(sic)
        industries_df['Food Products'] = sic_df[food_xs].sum(axis=1)
        industries_wrets_df['Food Products'] = sic_wrets_df[food_xs].sum(axis=1) 
    elif (2064 <= sic <= 2068) or (sic == 2086) or (sic == 2087) or (sic == 2096) or (sic == 2097): #3 soda
        soda_xs.append(sic)
        industries_df['Candy and Soda'] = sic_df[soda_xs].sum(axis=1)  
        industries_wrets_df['Candy and Soda'] = sic_wrets_df[soda_xs].sum(axis=1) 
    elif (sic == 2080) or (2082 <= sic <= 2085): #4 beer
        beer_xs.append(sic)
        industries_df['Beer and Liquor'] = sic_df[beer_xs].sum(axis=1) 
        industries_wrets_df['Beer and Liquor'] = sic_wrets_df[soda_xs].sum(axis=1) 
    elif (2100 <= sic <= 2199): #5 smoke
        smoke_xs.append(sic)
        industries_df['Tobacco Products'] = sic_df[smoke_xs].sum(axis=1)
        industries_wrets_df['Tobacco Products'] = sic_wrets_df[smoke_xs].sum(axis=1) 
    elif (920 <= sic <= 999) or (3650 <= sic <= 3652) or (sic == 3732) or (3930 <= sic <= 3931) or (3940 <= sic <= 3949): #6 toys
        toys_xs.append(sic)
        industries_df['Recreation'] = sic_df[toys_xs].sum(axis=1)
        industries_wrets_df['Recreation'] = sic_wrets_df[toys_xs].sum(axis=1) 
    elif ((7800 <= sic <= 7833) or (7840 <= sic <= 7841) or (sic == 7900) or (7910 <= sic <= 7911) or (7920 <= sic <= 7933)
        or (7940 <= sic <= 7949) or (sic == 7980) or (7990 <= sic <= 7999)): #7 fun
        fun_xs.append(sic)
        industries_df['Entertainment'] = sic_df[fun_xs].sum(axis=1)   
        industries_wrets_df['Entertainment'] = sic_wrets_df[fun_xs].sum(axis=1) 
    elif (2700 <= sic <= 2749) or (2770 <= sic <= 2771) or (2780 <= sic <= 2799): #8 books
        books_xs.append(sic)
        industries_df['Printing and Publishing'] = sic_df[books_xs].sum(axis=1)
        industries_wrets_df['Printing and Publishing'] = sic_wrets_df[books_xs].sum(axis=1)  
    elif ((sic == 2047) or (2391 <= sic <= 2392) or (2510 <= sic <= 2519) or (2590 <= sic <= 2599) or (2840 <= sic <= 2844)
        or (3160 <= sic <= 3161) or (3170 <= sic <= 3171) or (sic == 3172) or (3190 <= sic <= 3199) or (sic == 3229)
        or (sic == 3260) or (3262 <= sic <= 3263) or (sic == 3269) or (3230 <= sic <= 3231) or (3630 <= sic <= 3639) or
        (3750 <= sic <= 3751) or (sic == 3800) or (3860 <= sic <= 3861) or (3870 <= sic <= 3873) or (3910 <= sic <= 3911) or
        (3914 <= sic <= 3915) or (3960 <= sic <= 3962) or (sic == 3991) or (sic == 3995)): #9 hshld
        hshld_xs.append(sic)
        industries_df['Consumer Goods'] = sic_df[hshld_xs].sum(axis=1)   
        industries_wrets_df['Consumer Goods'] = sic_wrets_df[hshld_xs].sum(axis=1) 
    elif ((2300 <= sic <= 2390) or (3020 <= sic <= 3021) or (3100 <= sic <= 3111) or (3130 <= sic <= 3131) or 
    (3140 <= sic <= 3149) or (3150 <= sic <= 3151) or (3963 <= sic <= 3965)): #10 clths
        clths_xs.append(sic)
        industries_df['Apparel'] = sic_df[clths_xs].sum(axis=1)   
        industries_wrets_df['Apparel'] = sic_wrets_df[clths_xs].sum(axis=1)  
    elif (8000 <= sic <= 8099): #11 hlth
        hlth_xs.append(sic)
        industries_df['Healthcare'] = sic_df[hlth_xs].sum(axis=1)
        industries_wrets_df['Healthcare'] = sic_wrets_df[hlth_xs].sum(axis=1)
    elif (sic == 3693) or (3840 <= sic <= 3851): #12 medeq
        medeq_xs.append(sic)
        industries_df['Medical Equipment'] = sic_df[medeq_xs].sum(axis=1)
        industries_wrets_df['Medical Equipment'] = sic_wrets_df[medeq_xs].sum(axis=1)
    elif (2830 <= sic <= 2831) or (2833 <= sic <= 2836): #13 drugs
        drugs_xs.append(sic)
        industries_df['Pharmaceutical Products'] = sic_df[drugs_xs].sum(axis=1)
        industries_wrets_df['Pharmaceutical Products'] = sic_wrets_df[drugs_xs].sum(axis=1)
    elif (2800 <= sic <= 2829) or (2850 <= sic <= 2879) or (2890 <= sic <= 2899): #14 chems
        chems_xs.append(sic)
        industries_df['Chemicals'] = sic_df[chems_xs].sum(axis=1)
        industries_wrets_df['Chemicals'] = sic_wrets_df[chems_xs].sum(axis=1)
    elif (sic == 3031) or (sic == 3041) or (3050 <= sic <= 3053) or (3060 <= sic <= 3099): #15 rubbr
        rubbr_xs.append(sic)
        industries_df['Rubber and Plastic Products'] = sic_df[rubbr_xs].sum(axis=1)
        industries_wrets_df['Rubber and Plastic Products'] = sic_wrets_df[rubbr_xs].sum(axis=1)
    elif ((2200 <= sic <= 2284) or (2290 <= sic <= 2295) or (2297 <= sic <= 2299) or (2393 <= sic <= 2395)
        or (2397 <= sic <= 2399)): #16 txtls
        txtls_xs.append(sic)
        industries_df['Textiles'] = sic_df[txtls_xs].sum(axis=1)
        industries_wrets_df['Textiles'] = sic_wrets_df[txtls_xs].sum(axis=1)
    elif ((800 <= sic <= 899) or (2400 <= sic <= 2439) or (2450 <= sic <= 2459) or (2490 <= sic <= 2499)
        or (2660 <= sic <= 2661) or (2950 <= sic <= 2952) or (sic == 3200) or (3210 <= sic <= 3211)
        or (3240 <= sic <= 3241) or (3250 <= sic <= 3259) or (sic == 3261) or (sic == 3264) or (3270 <= sic <= 3275)
        or (3280 <= sic <= 3281) or (3290 <= sic <= 3293) or (3295 <= sic <= 3299) or (3420 <= sic <= 3429)
        or (3430 <= sic <= 3433) or (3440 <= sic <= 3441) or (sic == 3442) or (sic == 3446) or (sic == 3448)
        or (sic == 3449) or (3450 <= sic <= 3451) or (sic == 3452) or (3490 <= sic <= 3499) or (sic == 3996)): #17 bldmt
        bldmt_xs.append(sic)
        industries_df['Construction Materials'] = sic_df[bldmt_xs].sum(axis=1)
        industries_wrets_df['Construction Materials'] = sic_wrets_df[bldmt_xs].sum(axis=1)
    elif (1500 <= sic <= 1511) or (1520 <= sic <= 1549) or (1600 <= sic <= 1799): #18 cnstr
        cnstr_xs.append(sic)
        industries_df['Construction'] = sic_df[cnstr_xs].sum(axis=1)
        industries_wrets_df['Construction'] = sic_wrets_df[cnstr_xs].sum(axis=1)
    elif ((sic == 3300) or (3310 <= sic <= 3317) or (3320 <= sic <= 3325) or (3330 <= sic <= 3339)
        or (3340 <= sic <= 3341) or (3350 <= sic <= 3357) or (3360 <= sic <= 3379) or 
        (3390 <= sic <= 3399)): #19 steel
        steel_xs.append(sic)
        industries_df['Steel Works Etc'] = sic_df[steel_xs].sum(axis=1)
        industries_wrets_df['Steel Works Etc'] = sic_wrets_df[steel_xs].sum(axis=1)
    elif (sic == 3400) or (3443 <= sic <= 3444) or (3460 <= sic <= 3479): #20 fabpr
        fabpr_xs.append(sic)
        industries_df['Fabricated Products'] = sic_df[fabpr_xs].sum(axis=1)
        industries_wrets_df['Fabricated Products'] = sic_wrets_df[fabpr_xs].sum(axis=1)
    elif ((3510 <= sic <= 3536) or (sic == 3538) or (3540 <= sic <= 3569) or (3580 <= sic <= 3582)
        or (3585 <= sic <= 3586) or (3589 <= sic <= 3599)): #21 mach
        mach_xs.append(sic)
        industries_df['Machinery'] = sic_df[mach_xs].sum(axis=1)
        industries_wrets_df['Machinery'] = sic_wrets_df[mach_xs].sum(axis=1)
    elif ((sic == 3600) or (3610 <= sic <= 3613) or (3620 <= sic <= 3621) or (3623 <= sic <= 3629)
        or (3640 <= sic <= 3646) or (3648 <= sic <= 3649) or (sic == 3660) or (3690 <= sic <= 3692)
        or (sic == 3699)): #22 elceq
        elceq_xs.append(sic)
        industries_df['Electrical Equipment'] = sic_df[elceq_xs].sum(axis=1)
        industries_wrets_df['Electrical Equipment'] = sic_wrets_df[elceq_xs].sum(axis=1)
    elif ((sic == 2296) or (sic == 2396) or (3010 <= sic <= 3011) or (sic == 3537) or (sic == 3649)
        or (sic == 3694) or (sic == 3700) or (sic == 3710) or (sic == 3711) or (3713 <= sic <= 3716) 
        or (3790 <= sic <= 3792) or (sic == 3799)): #23 autos
        autos_xs.append(sic)
        industries_df['Automobiles and Trucks'] = sic_df[autos_xs].sum(axis=1)
        industries_wrets_df['Automobiles and Trucks'] = sic_wrets_df[autos_xs].sum(axis=1)
    elif (3720 <= sic <= 3721) or (3723 <= sic <= 3725) or (3728 <= sic <= 3729): #24 aero
        aero_xs.append(sic)
        industries_df['Aircraft'] = sic_df[aero_xs].sum(axis=1)
        industries_wrets_df['Aircraft'] = sic_wrets_df[aero_xs].sum(axis=1)
    elif (3730 <= sic <= 3731) or (3740 <= sic <= 3743): #25 ships
        ships_xs.append(sic)
        industries_df['Shipbuilding, Railroad Equipment'] = sic_df[ships_xs].sum(axis=1)  
        industries_wrets_df['Shipbuilding, Railroad Equipment'] = sic_wrets_df[ships_xs].sum(axis=1)  
    elif (3760 <= sic <= 3769) or (sic == 3795) or (3480 <= sic <= 3489): #26 guns
        guns_xs.append(sic)
        industries_df['Defense'] = sic_df[guns_xs].sum(axis=1)
        industries_wrets_df['Defense'] = sic_wrets_df[guns_xs].sum(axis=1)
    elif (1040 <= sic <= 1049): #27 gold
        gold_xs.append(sic)
        industries_df['Precious Metals'] = sic_df[gold_xs].sum(axis=1)
        industries_wrets_df['Precious Metals'] = sic_wrets_df[gold_xs].sum(axis=1)
    elif (1000 <= sic <= 1039) or (1050 <= sic <= 1119) or (1400 <= sic <= 1499): #28 mines
        mines_xs.append(sic)
        industries_df['Non-Metallic and Industrial Metal Mining'] = sic_df[mines_xs].sum(axis=1)
        industries_wrets_df['Non-Metallic and Industrial Metal Mining'] = sic_wrets_df[mines_xs].sum(axis=1)
    elif (1200 <= sic <= 1299): #29 coal
        coal_xs.append(sic)
        industries_df['Coal'] = sic_df[coal_xs].sum(axis=1)
        industries_wrets_df['Coal'] = sic_wrets_df[coal_xs].sum(axis=1)
    elif ((sic == 1300) or (1310 <= sic <= 1339) or (1370 <= sic <= 1382) or (sic == 1389) or 
        (2900 <= sic <= 2912) or (2990 <= sic <= 2999)): #30 oil
        oil_xs.append(sic)
        industries_df['Petroleum and Natural Gas'] = sic_df[oil_xs].sum(axis=1)
        industries_wrets_df['Petroleum and Natural Gas'] = sic_wrets_df[oil_xs].sum(axis=1)
    elif ((sic == 4900) or (4910 <= sic <= 4911) or (4920 <= sic <= 4925) or (4930 <= sic <= 4932) or 
        (sic == 4939) or (4940 <= sic <= 4942)): #31 util
        util_xs.append(sic)
        industries_df['Utilities'] = sic_df[util_xs].sum(axis=1)
        industries_wrets_df['Utilities'] = sic_wrets_df[util_xs].sum(axis=1)
    elif ((sic == 4800) or (4810 <= sic <= 4813) or (4820 <= sic <= 4822) or (4830 <= sic <= 4841)
        or (4880 <= sic <= 4892) or (sic == 4899)): #32 telcm
        telcm_xs.append(sic)
        industries_df['Communication'] = sic_df[telcm_xs].sum(axis=1)
        industries_wrets_df['Communication'] = sic_wrets_df[telcm_xs].sum(axis=1)
    elif ((7020 <= sic <= 7021) or (7030 <= sic <= 7033) or (sic == 7200) or (7210 <= sic <= 7212)
        or (7214 <= sic <= 7217) or (7219 <= sic <= 7221) or (7230 <= sic <= 7231) or (7240 <= sic <= 7241)
        or (7250 <= sic <= 7251) or (7260 <= sic <= 7299) or (sic == 7395) or (sic == 7500) or 
        (7520 <= sic <= 7549) or (sic == 7600) or (sic == 7620) or (sic == 7622) or (sic == 7623)
        or (7629 <= sic <= 7631) or (7640 <= sic <= 7641) or (7690 <= sic <= 7699) or (8100 <= sic <= 8499)
        or (8600 <= sic <= 8699) or (8800 <= sic <= 8899) or (7510 <= sic <= 7515)): #33 persv
        persv_xs.append(sic)
        industries_df['Personal Services'] = sic_df[persv_xs].sum(axis=1)
        industries_wrets_df['Personal Services'] = sic_wrets_df[persv_xs].sum(axis=1)
    elif ((2750 <= sic <= 2759) or (sic == 3993) or (sic == 7218) or (sic == 7300)
        or (7310 <= sic <= 7342) or (7349 <= sic <= 7353) or (7359 <= sic <= 7372) or (7374 <= sic <= 7385)
        or (7389 <= sic <= 7394) or (7396 <= sic <= 7397) or (sic == 7399) or (sic == 7519) or 
        (sic == 8700) or (8710 <= sic <= 8713) or (8720 <= sic <= 8721) or (8730 <= sic <= 8734) or (8740 <= sic <= 8748)
        or (8900 <= sic <= 8910) or (sic == 8911) or (8920 <= sic <= 8999) or (4220 <= sic <= 4229)): #34 bussv
        bussv_xs.append(sic)
        industries_df['Business Services'] = sic_df[bussv_xs].sum(axis=1)
        industries_wrets_df['Business Services'] = sic_wrets_df[bussv_xs].sum(axis=1)
    elif (3570 <= sic <= 3579) or (3680 <= sic <= 3689) or (sic == 3695) or (sic == 7373): #35 comps
        comps_xs.append(sic)
        industries_df['Computers'] = sic_df[comps_xs].sum(axis=1)
        industries_wrets_df['Computers'] = sic_wrets_df[comps_xs].sum(axis=1)
    elif (sic == 3622) or (3661 <= sic <= 3666) or (3669 <= sic == 3679) or (sic == 3810) or (sic == 3812): #36 chips
        chips_xs.append(sic)
        industries_df['Electronic Equipment'] = sic_df[chips_xs].sum(axis=1)
        industries_wrets_df['Electronic Equipment'] = sic_wrets_df[chips_xs].sum(axis=1)
    elif (sic <= 3811) or (3820 <= sic <= 3827) or (3829 <= sic == 3839): #37 labeq
        labeq_xs.append(sic)
        industries_df['Measuring and Control Equipment'] = sic_df[labeq_xs].sum(axis=1)
        industries_wrets_df['Measuring and Control Equipment'] = sic_wrets_df[labeq_xs].sum(axis=1)
    elif ((2520 <= sic <= 2549) or (2600 <= sic <= 2639) or (2670 <= sic == 2699) or (2760 <= sic <= 2761)
          or (3950 <= sic <= 3955)): #38 paper
        paper_xs.append(sic)
        industries_df['Business Supplies'] = sic_df[paper_xs].sum(axis=1)
        industries_wrets_df['Business Supplies'] = sic_wrets_df[paper_xs].sum(axis=1)
    elif (2440 <= sic <= 2449) or (2640 <= sic <= 2659) or (3220 <= sic <= 3221) or (3410 <= sic <= 3412): #39 boxes
        boxes_xs.append(sic)
        industries_df['Shipping Containers'] = sic_df[boxes_xs].sum(axis=1)
        industries_wrets_df['Shipping Containers'] = sic_wrets_df[boxes_xs].sum(axis=1)
    elif ((4000 <= sic <= 4013) or (4040 <= sic <= 4049) or (sic == 4100) or (4110 <= sic <= 4121)
          or (4130 <= sic <= 4131) or (4140 <= sic <= 4142) or (4150 <= sic <= 4151) or (4170 <= sic <= 4173)
          or (4190 <= sic <= 4200) or (4210 <= sic <= 4219) or (4230 <= sic <= 4231) or (4240 <= sic <= 4249)
          or (4400 <= sic <= 4499) or (4500 <= sic <= 4700) or (4710 <= sic <= 4712) or (4720 <= sic <= 4749)
          or (sic == 4780) or (4782 <= sic <= 4785) or (sic == 4789)): #40 trans
        trans_xs.append(sic)
        industries_df['Transportation'] = sic_df[trans_xs].sum(axis=1)
        industries_wrets_df['Transportation'] = sic_wrets_df[trans_xs].sum(axis=1)
    elif ((sic == 5000) or (5010 <= sic <= 5015) or (5020 <= sic <= 5023) or (5030 <= sic <= 5060)
          or (5063 <= sic <= 5065) or (5070 <= sic <= 5078) or (5080 <= sic <= 5088) or (5090 <= sic <= 5094)
          or (5099 <= sic <= 5100) or (5110 <= sic <= 5113) or (5120 <= sic <= 5122) or (5130 <= sic <= 5172)
          or (5180 <= sic <= 5182) or (5190 <= sic <= 5199)): #41 whlsl
        whlsl_xs.append(sic)
        industries_df['Wholesale'] = sic_df[whlsl_xs].sum(axis=1)
        industries_wrets_df['Wholesale'] = sic_wrets_df[whlsl_xs].sum(axis=1)
    elif ((sic == 5200) or (5210 <= sic <= 5231) or (5250 <= sic <= 5251) or (5260 <= sic <= 5261)
          or (5270 <= sic <= 5271) or (sic == 5300) or (5310 <= sic <= 5311) or (sic == 5320)
          or (5099 <= sic <= 5100) or (5110 <= sic <= 5113) or (5120 <= sic <= 5122) or (5130 <= sic <= 5172)
          or (5330 <= sic <= 5331) or (sic == 5334) or (5340 <= sic <= 5349) or (5390 <= sic <= 5400)
          or (5410 <= sic <= 5412) or (5420 <= sic <= 5469) or (5490 <= sic <= 5500) or (5510 <= sic <= 5579)
          or (5590 <= sic <= 5700) or (5710 <= sic <= 5722) or (5730 <= sic <= 5736) or (5750 <= sic <= 5799)
          or (sic == 5900) or (5910 <= sic <= 5912) or (5920 <= sic <= 5932) or (5940 <= sic <= 5990)
          or (5992 <= sic <= 5995) or (sic == 5999)): #42 rtail
        rtail_xs.append(sic)
        industries_df['Retail'] = sic_df[rtail_xs].sum(axis=1)
        industries_wrets_df['Retail'] = sic_wrets_df[rtail_xs].sum(axis=1)
    elif ((5800 <= sic <= 5829) or (5890 <= sic <= 5899) or (sic == 7000) or (7010 <= sic <= 7019)
          or (7040 <= sic <= 7049) or (sic == 7213)): #43 meals
        meals_xs.append(sic)
        industries_df['Restaurants, Hotels, Motels'] = sic_df[meals_xs].sum(axis=1)
        industries_wrets_df['Restaurants, Hotels, Motels'] = sic_wrets_df[meals_xs].sum(axis=1)
    elif ((sic == 6000) or (6010 <= sic <= 6036) or (6040 <= sic <= 6062) or (6080 <= sic <= 6082)
          or (6090 <= sic <= 6100) or (6110 <= sic <= 6113) or (6120 <= sic <= 6179) or (6190 <= sic <= 6199)): #44 banks
        banks_xs.append(sic)
        industries_df['Banking'] = sic_df[banks_xs].sum(axis=1)
        industries_wrets_df['Banking'] = sic_wrets_df[banks_xs].sum(axis=1)
    elif ((sic == 6300) or (6310 <= sic <= 6331) or (6350 <= sic <= 6351) or (6360 <= sic <= 6361)
          or (6370 <= sic <= 6379) or (6390 <= sic <= 6411)): #45 insur
        insur_xs.append(sic)
        industries_df['Insurance'] = sic_df[insur_xs].sum(axis=1)
        industries_wrets_df['Insurance'] = sic_wrets_df[insur_xs].sum(axis=1)
    elif ((sic == 6500) or (sic == 6510) or (6512 <= sic <= 6515) or (6517 <= sic <= 6532)
          or (6540 <= sic <= 6541) or (6550 <= sic <= 6553) or (6590 <= sic <= 6599) or (6610 <= sic <= 6611)): #46 rlest
        rlest_xs.append(sic)
        industries_df['Real Estate'] = sic_df[rlest_xs].sum(axis=1)
        industries_wrets_df['Real Estate'] = sic_wrets_df[rlest_xs].sum(axis=1)
    elif ((6200 <= sic <= 6299) or (sic == 6700) or (6710 <= sic <= 6726) or (6730 <= sic <= 6733)
          or (6740 <= sic <= 6779) or (6790 <= sic <= 6795) or (6798 <= sic <= 6799)): #47 fin
        fin_xs.append(sic)
        industries_df['Trading'] = sic_df[fin_xs].sum(axis=1)
        industries_wrets_df['Trading'] = sic_wrets_df[fin_xs].sum(axis=1)
    elif sic == 4950: #48 other
        other_xs.append(sic)
        industries_df['Sanitary Services'] = sic_df[other_xs].sum(axis=1)
        industries_wrets_df['Sanitary Services'] = sic_wrets_df[other_xs].sum(axis=1)
    elif sic == 9999: #49 nce
        nce_xs.append(sic)
        industries_df['Nonclassifiable Establishments'] = sic_df[nce_xs].sum(axis=1)
        industries_wrets_df['Nonclassifiable Establishments'] = sic_wrets_df[nce_xs].sum(axis=1)
    else:
        print(sic) # <-- checks for not yet grouped SIC codes
        
del(agric_xs, # <-- Clean variable explorer; lists no longer needed
food_xs,
soda_xs,
beer_xs,
smoke_xs,
toys_xs,
fun_xs,
books_xs,
hshld_xs,
clths_xs,
hlth_xs,
medeq_xs,
drugs_xs,
chems_xs,
rubbr_xs,
txtls_xs,
bldmt_xs,
cnstr_xs,
steel_xs,
fabpr_xs,
mach_xs,
elceq_xs,
autos_xs,
aero_xs,
ships_xs,
guns_xs,
gold_xs,
mines_xs,
coal_xs,
oil_xs,
util_xs,
telcm_xs,
persv_xs,
bussv_xs,
comps_xs,
chips_xs,
labeq_xs,
paper_xs,
boxes_xs,
trans_xs,
whlsl_xs,
rtail_xs,
meals_xs,
banks_xs,
insur_xs,
rlest_xs,
fin_xs,
other_xs)

industries_wrets_df.to_csv('FIN610-wRetsIndustry.csv')
industries_wrets_df.plot(figsize = (18,6)).set_title('Weighted Returns of All Positions')

industries_wcumrets_df = pd.DataFrame().reindex_like(industries_wrets_df)
industries_wcumrets_df = (1 + industries_wrets_df).cumprod() - 1
industries_wcumrets_df.to_csv('FIN610-weightedcumrets_Industry.csv')
industries_wcumrets_df.plot(figsize = (18,6)).set_title('Weighted Cumulative Returns of All Positions')

finalcumrets_industry_df = industries_wcumrets_df.ffill(axis=0).iloc[-1,:]
finalcumrets_industry_df.to_csv('FIN610-FinalCumRets_Industry.csv')

# industries_df.plot(figsize = (18,6)).set_title('4. Daily Industry Distribution of Holdings in Market Value From May 2010 to January 2014')
# plt.savefig('FIN610-Project Daily Industry Distribution of Holdings in Market Value From May 2010 to January 2014.png')

weights_industry_df = industries_df.div(industries_df.sum(axis=1), axis=0)
# weights_industry_df.plot(figsize = (18,6)).set_title('5. Daily Industry Distribution of Holdings in Percentage From May 2010 to January 2014')
# plt.savefig('FIN610-Project Daily Industry Distribution of Holdings in Percentage From May 2010 to January 2014.png')
weights_industry_df.to_csv('FIN610-weightsINDUSTRY.csv')


etf_df = ID_df[['ISHARES RUSS 2000 ETF',
                            'ISHARES RUSS 2000 GROWTH ETF',
                            'ISHARES NA NATURAL RESOURCES ETF',
                            'ISHARES US REAL ESTATE ETF',
                            'ISHARES CORE SNP SMALLCAP ETF',
                            'ISHARES RUSS 2000 VALUE ETF',
                            'ISHARES MSCI TURKEY ETF',
                            'ISHARES MSCI GERMANY ETF']] 

# weights_ID_df.plot(figsize = (18,6)).set_title('Weights of Each Position by ID')

weights_etfsamongetfs_df = etf_df.div(etf_df.sum(axis=1), axis=0) 
weights_etfsamongetfs_df.to_csv('FIN610-weightsETFsAmongETFs.csv')


stocksonly_pivot_df = ID_df.drop(['ISHARES RUSS 2000 ETF',
                                        'ISHARES RUSS 2000 GROWTH ETF',
                                        'ISHARES NA NATURAL RESOURCES ETF',
                                        'ISHARES US REAL ESTATE ETF',
                                        'ISHARES CORE SNP SMALLCAP ETF',
                                        'ISHARES RUSS 2000 VALUE ETF',
                                        'ISHARES MSCI TURKEY ETF',
                                        'ISHARES MSCI GERMANY ETF'],1)

weights_stocksamongstocks_df = stocksonly_pivot_df.div(stocksonly_pivot_df.sum(axis=1), axis=0) 
weights_stocksamongstocks_df.to_csv('FIN610-weightsStocksAmongStocks.csv')

mcap_df = positions_df.pivot_table(index = 'DATE', columns = 'ID', values = 'mcap', aggfunc='sum') # <-- Analysis of mcap size of individual equity positions

mcap_stocksonly_pivot_df = mcap_df.drop(['ISHARES RUSS 2000 ETF',
                                        'ISHARES RUSS 2000 GROWTH ETF',
                                        'ISHARES NA NATURAL RESOURCES ETF',
                                        'ISHARES US REAL ESTATE ETF',
                                        'ISHARES CORE SNP SMALLCAP ETF',
                                        'ISHARES RUSS 2000 VALUE ETF',
                                        'ISHARES MSCI TURKEY ETF',
                                        'ISHARES MSCI GERMANY ETF'],1)
mcap_stocksonly_pivot_df.to_csv('mcaps of stocks only.csv')


stocksonly_df = positions_df[positions_df['ID'].str.contains("ETF")==False]

"""
Categorized individual equity positions by 2014 IWM cutoff for Smallcap/large cap
In Thousands Small Cap    0-5,400,000
Large Cap 5,400,000-
"""

stocksonly_df['Size Classification'] = pd.cut(stocksonly_df['mcap'], bins=[0, 5400000, 10000000000], 
                                              include_lowest=True, labels=['Small Cap', 'Large Cap'])


largecapstocks_df = stocksonly_df.loc[stocksonly_df['Size Classification'] == 'Large Cap']
largecapstocks_df.to_csv('FIN610-Large Cap Stocks Dataframe.csv')

largecapstocks_xs = largecapstocks_df['ID'].unique() # <-- Large cap stocks outlier in portfolio, check listed positions

classification_df = stocksonly_df.pivot_table(index = 'DATE', columns = 'Size Classification', values = 'position_value', aggfunc='sum')

classification_df.columns = classification_df.columns.add_categories(['ISHARES RUSS 2000 ETF',
                                                                      'ISHARES RUSS 2000 GROWTH ETF',
                                                                      'ISHARES NA NATURAL RESOURCES ETF',
                                                                      'ISHARES US REAL ESTATE ETF',
                                                                      'ISHARES CORE SNP SMALLCAP ETF',
                                                                      'ISHARES RUSS 2000 VALUE ETF',
                                                                      'ISHARES MSCI TURKEY ETF',
                                                                      'ISHARES MSCI GERMANY ETF']) 
classification_df['ISHARES RUSS 2000 ETF'] = ID_df['ISHARES RUSS 2000 ETF']
classification_df['ISHARES RUSS 2000 GROWTH ETF'] = ID_df['ISHARES RUSS 2000 GROWTH ETF']
classification_df['ISHARES NA NATURAL RESOURCES ETF'] = ID_df['ISHARES NA NATURAL RESOURCES ETF']
classification_df['ISHARES US REAL ESTATE ETF'] = ID_df['ISHARES US REAL ESTATE ETF']
classification_df['ISHARES CORE SNP SMALLCAP ETF'] = ID_df['ISHARES CORE SNP SMALLCAP ETF']
classification_df['ISHARES RUSS 2000 VALUE ETF'] = ID_df['ISHARES RUSS 2000 VALUE ETF']
classification_df['ISHARES MSCI TURKEY ETF'] = ID_df['ISHARES MSCI TURKEY ETF']
classification_df['ISHARES MSCI GERMANY ETF'] = ID_df['ISHARES MSCI GERMANY ETF']

weights_classification_df = classification_df.div(classification_df.sum(axis=1), axis=0) 
weights_classification_df.to_csv('FIN610-weightsClassification.csv')

weights_classification_df.plot(figsize = (18,6)).set_title('Weights of Smallcap Stocks And ETF Types')
plt.axhline(y=0, color='k', linestyle='-')
plt.legend(loc = 'upper left')
plt.savefig('FIN610-Plot of Weights of Smallcap Stocks And ETF Types.png')

weights_stocksVSetfs_df = weights_industry_df['Trading'].to_frame().rename(columns = {'Trading':'ETFs'})
weights_stocksVSetfs_df['Stocks'] = 1 - weights_industry_df['Trading']
# weights_stocksVSetfs_df.plot(figsize = (18,6)).set_title('Weights of Stocks vs ETFs')
# plt.savefig('FIN610-Plot of Weights of Stocks vs ETFs')
weights_stocksVSetfs_df.to_csv('FIN610-weightsStocksVsETFs.csv')

### Analysis of Average Position Size Over Time

"""
Average position size 3% np.nanmean(weights_ID_df)
Median position size 3.26% np.nanmedian(weights_ID_df)
"""
avgweight_ID_df = pd.DataFrame(index=weights_ID_df.index)
avgweight_ID_df['mean'] = weights_ID_df.mean(axis=1).to_frame()
avgweight_ID_df['median'] = weights_ID_df.median(axis=1).to_frame()
avgweight_ID_df.to_csv('FIN610-avg weight by ID.csv')
avgweight_ID_df.plot(figsize = (18,6)).set_title('Mean and Median Weights of Individual Positions from 2010 to 2014')
plt.savefig('FIN610-Mean and Median Weights of Individual Positions from 2010 to 2014')




"""
find definition of value/growth
crsp -> holdings table, find securities in each ETF and match them to stocks
"""

### Analysis of position duration (defined as number of consecutive days with non-nan net_shares)

counter = 0
day = 0
ID_xs = []

for ID in ID_df:
    day = 0
    for position in ~ID_df[ID].isnull():
        day += 1
        if position == True:
            counter += 1
        if (position == False and counter != 0) or (day == len(ID_df) and counter != 0):
        #     globals()[ticker + "_xs"].append(counter)
            ID_xs.append(counter)
            counter = 0   
            
"""
Iterate through each value in the dataframe, checking for nan values (bitwise negation operator of isnull())
    Once a non-nan value is found, add to counter until a nan value is reached, 
    at which point check if counter contains a value greater than zero (starts at 0) and if so, 
    store the result in a list containing the output
    Also always count the row number of the list to check if the iterated value is at the last row of df, 
    at which point the counter result will also be appended assuming it is > 0    
"""

position_duration_df = pd.DataFrame(data=ID_xs)
position_duration_df.to_csv('FIN610-Duration of Positions.csv')

position_duration_df.plot.bar(figsize = (18,6), legend=None
    ).set_title('Position Duration in Days from May 2010 to January 2014').axes.get_xaxis().set_visible(False)

position_duration_df.plot.hist(figsize = (18,6), legend=None, align = 'mid', 
                               bins = [0,50,100,150,200,250,300,350,400,450,500,550,600,
                                       650,700,750,800,850,900,1000]
    ).set_title('Distribution of Position Duration in Days from May 2010 to January 2014')
plt.xticks(np.arange(0, 1000, 50))
plt.savefig('FIN610-Histogram of Position Durations')

print(f'Average duration of position = {np.mean(ID_xs)} days')
print(f'Median duration of position = {np.median(ID_xs)} days')


### Performance Assessment ###

### Incorporate weights with returns



#check value close with portfolio values # <--


"""
Next:
Performance assessment

incorporate returns with weights
incorporate std with weights
"""

# 2. Portfolio dataframe

portfolio_df = pd.read_csv('Portfolio_2021.csv',
                      index_col = 'DATE',
                      parse_dates = True).drop(['number_short',
                                                'number_long',
                                                'lev_multiple'],1)
                                                
portfolio_df = portfolio_df.loc['2010-05-03':] # <-- Begin data at May 5, 2010 per project guidelines
                                                
portfolio_df['CumRet2 Check'] = (1 + weightedrets_ID_df.sum(axis=1)).cumprod() - 1
portfolio_df['CumRet2'] = (1 + portfolio_df['ret2']).cumprod() - 1
portfolio_df['CumRet3'] = (1 + portfolio_df['ret3']).cumprod() - 1

portfolio_df.plot(figsize = (18,6), y = ['CumRet2 Check', 'CumRet2', 'CumRet3']).set_title(
                   'Portfolio CumRets')
plt.savefig('FIN610-Plot of CumRets Check')

"""
ret2 = daily gain of fund's position assuming 100% collateral posted for short positions
value_close = sum of absolute values of security positions at the end of the day (as in ret2)
ret3 = daily gain of fund's position assuming 0% collateral posted for short positions
ret2_long = daily gain of long side only (rebalanced)
value_close_long = sum of values of long positions at the end of the day
ret2_short = daily gain of short side only (rebalanced)
value_close_short = sum of values of short positions at the end of the day

*the net of these long and short values can be used as a base for ret3 weights
"""

print(" --- %s seconds ---" % (time.time() - start_time))