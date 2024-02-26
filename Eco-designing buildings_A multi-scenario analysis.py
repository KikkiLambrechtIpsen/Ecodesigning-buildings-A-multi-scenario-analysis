#!/usr/bin/env python
# coding: utf-8

# # Eco-designing buildigns: A multi-scenario analysis

# # 1. Set-up: Ecoinvent and World+ import

# In[1]:


# Import relevant packages to run the script
import pandas as pd
import numpy as np
import brightway2 as bw
from lci_to_bw2 import * # import all the functions of this module
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols


# In[2]:


bw.projects.set_current('PAPER_3_test_ecoinvent') # Still working in the same project
#bw.databases.clear() # For a fresh start (Risky command! clears all your existing databases)
#bw.databases


# In[3]:


# Import the biosphere3 database
bw.bw2setup()  # This will take some time


# Import of ecoinvent - Note that a licence is needed to get the relevant files

# In[4]:


# Import ecoinvent

# You need to change the line below with the directory where you have saved ecoinvent
ei38dir = "/Users/kikki/Sync/50-Kikki Ipsen/14_Databases/ecoinvent 3.8_consequential/datasets"

if 'ecoinvent 3.8 conseq' in bw.databases:
    print("Database has already been imported")
else:
    ei38 = bw.SingleOutputEcospold2Importer(ei38dir, 'ecoinvent 3.8 conseq') # You can give it another name of course
    ei38.apply_strategies()
    ei38.statistics()
    ei38.drop_unlinked(i_am_reckless=True)# an error in version 3.7 and 3.8 will be fixed in verison 3.9!
    ei38.write_database() # This will take some time.


# In[5]:


bw.databases # you should now see both "biosphere3" and "ecoinvent 3.8 conseq"


# In the following cell the impact assessment method Impact World+ is imported. The file IW_bw2.BW2PACKAGE is from:
# 
# laurepatou. (2019). laurepatou/IMPACT-World-in-Brightway: IW in Brightway: Midpoint 1.28 and Damage 1.46 (1.28_1.46). Zenodo. https://doi.org/10.5281/zenodo.3521041

# In[6]:


# Import of the World+ impact assessment method
#Note: Validate that there are no duplicates in the endpoint flows
import bw2io
imp = bw2io.package.BW2Package.load_file('IW_bw2.BW2PACKAGE', whitelist=True)

# Extract the methods from the import variable and detail the content 
for methods in imp :
    new_method = methods['name']
    new_cfs = methods['data']
    new_metadata = methods['metadata']
    my_method = bw.Method(new_method)
    my_method.register(**new_metadata)
    my_method.write(new_cfs)

# Validate that everything went well by checking one of the last methods
my_method_as_list1=[]
for ef in my_method.load() :
    my_method_as_list1.append([bw.get_activity(ef[0])['name'],bw.get_activity(ef[0])['categories'],
                                     ef[1]])    
df = pd.DataFrame(my_method_as_list1)
df.to_excel('Characterization method_detailed_3.xlsx', header = ('Elementary flow',"Emission locations", 
                                                       'Characterization factor'))

# Update the list of methods and validate that they are all present 
list_methods = []
for objects in bw.methods :
        list_methods.append(objects)

dataframe_methods = pd.DataFrame(data = list_methods, columns =('Method name',"Impact type",'Details') )
   
dataframe_methods.to_excel('Methodes.xlsx')


# # 2. Import inventory and conduct impact assessment

# Note that this part is iterative - it needs to be repeated until all needed impact assessments are done. 
# Use "iteration file_impact assessment (part 2 in script)".

# In[36]:


mydb = pd.read_csv('Database_y1.csv', header = 0, sep = ";") # using csv file avoids encoding problem
mydb.head()


# In[37]:


# Create a dict that can be written as database
bw2_db = lci_to_bw2(mydb) # a function from the lci_to_bw2 module
bw2_db


# In[38]:


t_db1 = bw.Database('y1_database') # it works because the database name in the excel file is the same
# shut down all other notebooks using the same project
t_db1.write(bw2_db)


# In[39]:


#Every time there is a number it needs to be the same as the inventory database that the impact assessement is done for
method = ['IMPACTWorld+ (Default_Recommended_Midpoint 1.28)']
category = ['Climate change, short term','Fossil and nuclear energy use', 'Mineral resources use',
            'Photochemical oxidant formation', 'Ozone Layer Depletion','Freshwater ecotoxicity', 'Human toxicity cancer',
            'Human toxicity non cancer','Freshwater acidification','Terrestrial acidification',
            'Freshwater eutrophication','Marine eutrophication','Land transformation, biodiversity',
            'Land occupation, biodiversity','Particulate matter formation', 'Ionizing radiations']

# Adjust this to the imported inventory database
housing = [t_db1.get('PAL6_total'), t_db1.get('a_total'), t_db1.get('b_total'), t_db1.get('c_total'), t_db1.get('d_total'),
t_db1.get('e_total'), t_db1.get('f_total'), t_db1.get('g_total'), t_db1.get('h_total')]

result=[]
for (a, b) in itertools.zip_longest(method, category, fillvalue='IMPACTWorld+ (Default_Recommended_Midpoint 1.28)'):
    method=(a,b)
    for h in housing [0:len(housing)]:
        house = h
        lca = bw.LCA({house:1}, method)
        lca.lci()
        lca.lcia()
        result.append(lca.score)

# The results from the impact assessment come in the form of a list and we want to make that into a matrix       
liste = result   
matrix = []
while liste != []:
    matrix.append(liste[:len(housing)])
    liste = liste[len(housing):]

# Convert the python matrix into a numpy array  
arr = np.array(matrix)
# Transpose the numpy array from impact categories X scenarios to scenarios X impact categories
arr_transpose = arr.transpose()

# Convert the numpy array to a pandas dataframe 
idx = ['PAL6', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
df1 = pd.DataFrame(arr_transpose, index = idx, columns = category)
df1.to_excel(r'C:\Users\kikki\Sync\50-Kikki Ipsen\y1_results.xlsx',index = True, header = True)
df1


# # 3. Creating the result matrix

# In[40]:


df_tot_12 = pd.DataFrame(
    pd.concat([df2 + n for n in df1.values]).values,
    index=pd.MultiIndex.from_product([df1.index, df2.index]), columns = df1.columns)


# In[41]:


df_tot_123 = pd.DataFrame(
    pd.concat([df3 + n for n in df_tot_12.values]).values,
    index=pd.MultiIndex.from_product([df1.index, df2.index, df3.index]), columns = df1.columns)


# In[42]:


df_tot_1234 = pd.DataFrame(
    pd.concat([df4 + n for n in df_tot_123.values]).values,
    index=pd.MultiIndex.from_product([df1.index, df2.index, df3.index, df4.index]), columns = df1.columns)


# In[43]:


df_tot_12345 = pd.DataFrame(
    pd.concat([df5 + n for n in df_tot_1234.values]).values,
    index=pd.MultiIndex.from_product([df1.index,df2.index,df3.index,df4.index,df5.index]), columns = df1.columns)


# In[44]:


df_tot_123456= pd.DataFrame(
    pd.concat([df6 + n for n in df_tot_12345.values]).values,
    index=pd.MultiIndex.from_product([df1.index,df2.index,df3.index,df4.index,df5.index,df6.index]), columns = df1.columns)


# In[45]:


df_tot_1234567= pd.DataFrame(
    pd.concat([df7 + n for n in df_tot_123456.values]).values,
    index=pd.MultiIndex.from_product([df1.index,df2.index,df3.index,df4.index,df5.index,df6.index,df7.index]), columns = df1.columns)


# In[46]:


col = ['A','B','S','D','E','F','G','H','I','J','K','L','M','N','O','P']
df_r= pd.DataFrame(
    pd.concat([df8 + n for n in df_tot_1234567.values]).values,
    index=pd.MultiIndex.from_product([df1.index,df2.index,df3.index,df4.index,df5.index,df6.index,df7.index,df8.index]), columns = col)
df_r


# In[47]:


df_result = df_r.reset_index()
df_result.columns = ['y1','y2','y3','y4','y5','y6','y7','y8','A','B','S','D','E','F','G','H','I','J','K','L','M','N','O','P']
df_result


# In[ ]:


df_result.to_csv(r'C:\Users\kikki\Sync\50-Kikki Ipsen\building_scenarios_impacts_total_v2.csv', sep=';', index = True, header = True)


# # 4. Plotting the distribution of the results

# Note that this part is iterative - it needs to be repeated until the distribution have been plotted for all assessed impact categories.Use "iteration file_distribution (part 4 in script)".

# In[ ]:


#Find the impact for the original design of PAL6, and the alternative designs a, b, etc.

AD_PAL6 = df_result[(df_result.y1=='PAL6') & (df_result.y2=='PAL6') & (df_result.y3=='PAL6') & (df_result.y4=='PAL6') &
                    (df_result.y5=='PAL6') & (df_result.y6=='PAL6') & (df_result.y7=='PAL6') & (df_result.y8=='PAL6')]

AD_a = df_result[(df_result.y1=='a') & (df_result.y2=='a') & (df_result.y3=='a') & (df_result.y4=='a') &
                    (df_result.y5=='PAL6') & (df_result.y6=='a') & (df_result.y7=='a') & (df_result.y8=='a')]

AD_b = df_result[(df_result.y1=='b') & (df_result.y2=='b') & (df_result.y3=='b') & (df_result.y4=='b') &
                    (df_result.y5=='b') & (df_result.y6=='b') & (df_result.y7=='PAL6') & (df_result.y8=='PAL6')]

AD_c = df_result[(df_result.y1=='c') & (df_result.y2=='c') & (df_result.y3=='c') & (df_result.y4=='c') &
                    (df_result.y5=='c') & (df_result.y6=='c') & (df_result.y7=='c') & (df_result.y8=='c')]

AD_d = df_result[(df_result.y1=='d') & (df_result.y2=='d') & (df_result.y3=='d') & (df_result.y4=='d') &
                    (df_result.y5=='d') & (df_result.y6=='d') & (df_result.y7=='PAL6') & (df_result.y8=='PAL6')]

AD_e = df_result[(df_result.y1=='e') & (df_result.y2=='e') & (df_result.y3=='e') & (df_result.y4=='e') &
                    (df_result.y5=='PAL6') & (df_result.y6=='e') & (df_result.y7=='a') & (df_result.y8=='a')]

AD_f = df_result[(df_result.y1=='f') & (df_result.y2=='f') & (df_result.y3=='f') & (df_result.y4=='f') &
                    (df_result.y5=='b') & (df_result.y6=='b') & (df_result.y7=='PAL6') & (df_result.y8=='PAL6')]

AD_g = df_result[(df_result.y1=='g') & (df_result.y2=='g') & (df_result.y3=='g') & (df_result.y4=='g') &
                    (df_result.y5=='c') & (df_result.y6=='g') & (df_result.y7=='c') & (df_result.y8=='c')]

AD_h = df_result[(df_result.y1=='h') & (df_result.y2=='d') & (df_result.y3=='h') & (df_result.y4=='d') &
                    (df_result.y5=='h') & (df_result.y6=='d') & (df_result.y7=='h') & (df_result.y8=='h')]


# In[ ]:


AD_PAL6 = AD_PAL6.iloc[0]
AD_a = AD_a.iloc[0]
AD_b = AD_b.iloc[0]
AD_c = AD_c.iloc[0]
AD_d = AD_d.iloc[0]
AD_e = AD_e.iloc[0]
AD_f = AD_f.iloc[0]
AD_g = AD_g.iloc[0]
AD_h = AD_h.iloc[0]


# In[ ]:


# Plot histogram
#define the color palette
color=sns.cubehelix_palette(start=2, rot=0, dark=.2, light=.6, reverse=True,)
sns.set_palette(color)
#set the style of the plot
sns.set_style("ticks")
# make the actuale plot
hist = sns.histplot(x=df_result["P"], stat='percent', bins=100, kde=True, zorder=1)
sns.despine()
# Add a vertical lines to the histogram and add text to explain what the line represents
plt.axvline(x=AD_PAL6[23], color='black', zorder=0)
plt.text(AD_PAL6[23], 19.5, "PAL6", horizontalalignment='center', size='medium', color='black', weight='semibold')

plt.axvline(x=AD_a[23], color='darkblue', zorder=0)
plt.text(AD_a[23], 19.5, "a", horizontalalignment='right', size='medium', color='darkblue', weight='semibold')

plt.axvline(x=AD_b[23], color='maroon', zorder=0)
plt.text(AD_b[23], 19.5, "b", horizontalalignment='center', size='medium', color='maroon', weight='semibold')

plt.axvline(x=AD_c[23], color='orange', zorder=0)
plt.text(AD_c[23], 19.5, "c", horizontalalignment='center', size='medium', color='orange', weight='semibold')

plt.axvline(x=AD_d[23], color='pink', zorder=0)
plt.text(AD_d[23], 19.5, "d", horizontalalignment='center', size='medium', color='pink', weight='semibold')

plt.axvline(x=AD_e[23], color='red', zorder=0)
plt.text(AD_e[23]+5e6, 19.5, "e", horizontalalignment='left', size='medium', color='red', weight='semibold')

plt.axvline(x=AD_f[23], color='lightblue', zorder=0)
plt.text(AD_f[23]+5e6, 19.5, "f", horizontalalignment='left', size='medium', color='lightblue', weight='semibold')

plt.axvline(x=AD_g[23], color='grey', zorder=0)
plt.text(AD_g[23]-5e6, 19.5, "g", horizontalalignment='right', size='medium', color='grey', weight='semibold')

plt.axvline(x=AD_h[23], color='green', zorder=0)
plt.text(AD_h[23]+3e7, 19.5, "h", horizontalalignment='left', size='medium', color='green', weight='semibold')

hist.set_xlabel("Ionizing radiations [Bq C-14 eq.]")
plt.savefig('distribution_Ionizing radiations.png')


# # 5. Regression

# Note that this part is iterative - it needs to be repeated until the regression analysis have been done for all assessed impact categories. Use "iteration file_regression (part 5 in script)".

# In[ ]:


# Replacing the categorical data, PAL6, a, b, c´, etc. with dummy values
df_result['y1PAL6'] = df_result.y1.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y1a'] = df_result.y1.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y1b'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y1c'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y1d'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y1e'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':1, 'f':0, 'g':0, 'h':0})
df_result['y1f'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':1, 'g':0, 'h':0})
df_result['y1g'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':1, 'h':0})
df_result['y1h'] = df_result.y1.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':1})

df_result['y2PAL6'] = df_result.y2.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y2a'] = df_result.y2.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y2b'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y2c'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y2d'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y2e'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':1, 'f':0, 'g':0, 'h':0})
df_result['y2f'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':1, 'g':0, 'h':0})
df_result['y2g'] = df_result.y2.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':1, 'h':0})

df_result['y3PAL6'] = df_result.y3.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y3a'] = df_result.y3.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y3b'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y3c'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y3d'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y3e'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':1, 'f':0, 'g':0, 'h':0})
df_result['y3f'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':1, 'g':0, 'h':0})
df_result['y3g'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':1, 'h':0})
df_result['y3h'] = df_result.y3.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':1})

df_result['y4PAL6'] = df_result.y4.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y4a'] = df_result.y4.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y4b'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y4c'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y4d'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y4e'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':1, 'f':0, 'g':0, 'h':0})
df_result['y4f'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':1, 'g':0, 'h':0})
df_result['y4g'] = df_result.y4.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':1, 'h':0})

df_result['y5PAL6'] = df_result.y5.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y5b'] = df_result.y5.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y5c'] = df_result.y5.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y5d'] = df_result.y5.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y5h'] = df_result.y5.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':1})

df_result['y6PAL6'] = df_result.y6.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y6a'] = df_result.y6.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y6b'] = df_result.y6.map({'PAL6':0, 'a':0, 'b':1, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y6c'] = df_result.y6.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y6d'] = df_result.y6.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':1, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y6e'] = df_result.y6.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':1, 'f':0, 'g':0, 'h':0})
df_result['y6g'] = df_result.y6.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':1, 'h':0})

df_result['y7PAL6'] = df_result.y7.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y7a'] = df_result.y7.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y7c'] = df_result.y7.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y7h'] = df_result.y7.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':1})

df_result['y8PAL6'] = df_result.y8.map({'PAL6':1, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y8a'] = df_result.y8.map({'PAL6':0, 'a':1, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y8c'] = df_result.y8.map({'PAL6':0, 'a':0, 'b':0, 'c':1, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0})
df_result['y8h'] = df_result.y8.map({'PAL6':0, 'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':1})

df_result


# In[ ]:


#Make the regression analysis - use the iteration file
fit = ols('P ~ C(y1) + C(y2) + C(y3) + C(y4) + C(y5) + C(y6) + C(y7) + C(y8)', data=df_result).fit()

with open('Regression_Ionizing radiations.txt', 'w') as fh:
    fh.write(fit.summary().as_text())

print(fit.summary())


# # 6. Filtering the results matrix

# ## 6.1 The best/worst compromise

# ### 6.1.1 Best compromise

# In[50]:


df_best = df_r.quantile(0.39)
df_best


# In[51]:


df_filt_min = df_r[(df_r.A<df_best[0]) & 
                    (df_r.B<df_best[1]) &
                    (df_r.S<df_best[2]) &
                    (df_r.D<df_best[3]) &
                    (df_r.E<df_best[4]) &
                    (df_r.F<df_best[5]) &
                    (df_r.G<df_best[6]) &
                    (df_r.H<df_best[7]) &
                    (df_r.I<df_best[8]) &
                    (df_r.J<df_best[9]) &
                    (df_r.K<df_best[10]) &
                    (df_r.L<df_best[11]) &
                    (df_r.M<df_best[12]) &
                    (df_r.N<df_best[13]) &
                    (df_r.O<df_best[14]) &
                    (df_r.P<df_best[15]) ]
df_filt_min


# In[ ]:


#Export this to a csv file
df_filt_min.to_csv(r'C:\Users\kikki\Sync\50-Kikki Ipsen\optimal solution.csv', sep=';', index = True, header = True)


# ### 6.1.2 worst compromise

# In[ ]:


df_worst = df_r.quantile(0.70)
df_worst


# In[ ]:


df_filt_max = df_r[(df_r.A>df_worst[0]) & 
                    (df_r.B>df_worst[1]) &
                    (df_r.S>df_worst[2]) &
                    (df_r.D>df_worst[3]) &
                    (df_r.E>df_worst[4]) &
                    (df_r.F>df_worst[5]) &
                    (df_r.G>df_worst[6]) &
                    (df_r.H>df_worst[7]) &
                    (df_r.I>df_worst[8]) &
                    (df_r.J>df_worst[9]) &
                    (df_r.K>df_worst[10]) &
                    (df_r.L>df_worst[11]) &
                    (df_r.M>df_worst[12]) &
                    (df_r.N>df_worst[13]) &
                    (df_r.O>df_worst[14]) &
                    (df_r.P>df_worst[15]) ]
df_filt_max


# In[ ]:


#Export this to a csv file
df_filt_max.to_csv(r'C:\Users\kikki\Sync\50-Kikki Ipsen\worst solution.csv', sep=';', index = True, header = True)


# ## 6.2 Plotting the best/worst compromise

# ### 6.2.1 Distribution 

# Note that this part is iterative - it needs to be repeated until the distribution of the optimal/worst-case solution have been plottet for all assessed impact categories. Use "Iteration file_best-worst distribution (part 6.2.1 in script)".

# In[ ]:


#set the style of the plot
sns.set_style("ticks")

# make the actuale plot
box=sns.boxplot(x=df_r["P"], width=0.2, color="white")
sns.stripplot(x=df_filt_min["P"], color="darkgreen")
sns.stripplot(x=df_filt_max["P"], color="maroon")
sns.despine(trim=True)

# add a vertical line to the histogram
plt.axvline(x=df_best[15], color='darkgreen')
plt.axvline(x=df_worst[15], color='maroon')
# Add text to explain what the line represents

plt.text(-1.8e8, -0.15, '39th percentile', size='medium', color='darkgreen', weight='semibold')
plt.text(0.4e8, -0.15, '70th percentile', size='medium', color='maroon', weight='semibold')
box.set_xlabel("Ionizing radiations [Bq C-14 eq.]")
plt.savefig('best-worst_Ionizing radiations.png')


# ### 6.2.2 Composition of best/worst compromise

# In[ ]:


db_count_min = pd.read_csv('optimal_solution_counted.csv', header = 0, sep = ";") # using csv file avoids encoding problem
db_count_min.head()


# In[ ]:


ind = ['PAL6','a','b','c','d','e','f','g', 'h']
dfcount_min = pd.DataFrame(pd.concat([db_count_min]).values, index=ind, columns = db_count_min.columns)
#dfcount_min


# In[ ]:


db_count_max = pd.read_csv('worst_solution_counted.csv', header = 0, sep = ";") # using csv file avoids encoding problem
db_count_max.head()


# In[ ]:


ind = ['PAL6','a','b','c','d','e','f','g', 'h']
dfcount_max = pd.DataFrame(pd.concat([db_count_max]).values, index=ind, columns = db_count_max.columns)
#dfcount_max


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(20,7))

color=sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, as_cmap=True)
sns.heatmap(dfcount_min, annot=True, cmap=color, fmt=".2f", vmin=0, vmax=4, linewidths=0.5, ax=axes[0])
sns.heatmap(dfcount_max, annot=True, cmap=color, fmt=".2f", vmin=0, vmax=4, linewidths=0.5, ax=axes[1])

fig.savefig('fig_4_count.png')


# # 7. Optimizing one impact categories

# Note that this part is iterative - it needs to be repeated until the 0,01% building scenarios with the smallest impact have been identified for each assessed impact categories. Use "Iteration file_optimize one impact category (part 7 in script)".

# In[ ]:


df_result.min()


# In[ ]:


one_filt = df_r.quantile(0.0001)
one_filt


# In[ ]:


#minimizing one impact category at the time
df_filt_one_min = df_result[(df_result.P<-5.555879e+08)]

df_filt_one_min.to_csv(r'C:\Users\kikki\Sync\50-Kikki Ipsen\optimize_one_Ionizing radiations.csv', sep=';', index = True, header = True)
len(df_filt_one_min)


# In[ ]:


fig, axes = plt.subplots(4,4, figsize=(16,10), constrained_layout = True)

#set the style of the plot
sns.set_style("ticks")

# make the actuale plot
boxA=sns.boxplot(x=df_result["A"], width=0.2, color="white", ax=axes[0,0])
sns.stripplot(x=df_filt_one_min["A"], color="darkgreen", ax=axes[0,0])

boxB=sns.boxplot(x=df_result["B"], width=0.2, color="white", ax=axes[0,1])
sns.stripplot(x=df_filt_one_min["B"], color="darkgreen", ax=axes[0,1])

boxC=sns.boxplot(x=df_result["S"], width=0.2, color="white", ax=axes[0,2])
sns.stripplot(x=df_filt_one_min["S"], color="darkgreen", ax=axes[0,2])

boxD=sns.boxplot(x=df_result["D"], width=0.2, color="white", ax=axes[0,3])
sns.stripplot(x=df_filt_one_min["D"], color="darkgreen", ax=axes[0,3])

boxE=sns.boxplot(x=df_result["E"], width=0.2, color="white", ax=axes[1,0])
sns.stripplot(x=df_filt_one_min["E"], color="darkgreen", ax=axes[1,0])

boxF=sns.boxplot(x=df_result["F"], width=0.2, color="white", ax=axes[1,1])
sns.stripplot(x=df_filt_one_min["F"], color="darkgreen", ax=axes[1,1])

boxG=sns.boxplot(x=df_result["G"], width=0.2, color="white", ax=axes[1,2])
sns.stripplot(x=df_filt_one_min["G"], color="darkgreen", ax=axes[1,2])

boxH=sns.boxplot(x=df_result["H"], width=0.2, color="white", ax=axes[1,3])
sns.stripplot(x=df_filt_one_min["H"], color="darkgreen", ax=axes[1,3])

boxI=sns.boxplot(x=df_result["I"], width=0.2, color="white", ax=axes[2,0])
sns.stripplot(x=df_filt_one_min["I"], color="darkgreen", ax=axes[2,0])

boxJ=sns.boxplot(x=df_result["J"], width=0.2, color="white", ax=axes[2,1])
sns.stripplot(x=df_filt_one_min["J"], color="darkgreen", ax=axes[2,1])

boxK=sns.boxplot(x=df_result["K"], width=0.2, color="white", ax=axes[2,2])
sns.stripplot(x=df_filt_one_min["K"], color="darkgreen", ax=axes[2,2])

boxL=sns.boxplot(x=df_result["L"], width=0.2, color="white", ax=axes[2,3])
sns.stripplot(x=df_filt_one_min["L"], color="darkgreen", ax=axes[2,3])

boxM=sns.boxplot(x=df_result["M"], width=0.2, color="white", ax=axes[3,0])
sns.stripplot(x=df_filt_one_min["M"], color="darkgreen", ax=axes[3,0])

boxN=sns.boxplot(x=df_result["N"], width=0.2, color="white", ax=axes[3,1])
sns.stripplot(x=df_filt_one_min["N"], color="darkgreen", ax=axes[3,1])

boxO=sns.boxplot(x=df_result["O"], width=0.2, color="white", ax=axes[3,2])
sns.stripplot(x=df_filt_one_min["O"], color="darkgreen", ax=axes[3,2])

boxP=sns.boxplot(x=df_result["P"], width=0.2, color="white", ax=axes[3,3])
sns.stripplot(x=df_filt_one_min["P"], color="darkgreen", ax=axes[3,3])

#Add the right label to the plot
boxA.set_xlabel("Climate change [kg CO2-eq.]")
boxB.set_xlabel("Fossil and nuclear energy use [MJ-deprived]")
boxC.set_xlabel("Mineral resources use [kg-deprived]")
boxD.set_xlabel("Photochemical oxidant formation [kg NMVOC-eq.]")
boxE.set_xlabel("Ozone Layer Depletion [kg CFC11-eq.]")
boxF.set_xlabel("Freshwater ecotoxicity [CTUe]")
boxG.set_xlabel("Human toxicity cancer [CTUh]")
boxH.set_xlabel("Human toxicity non cancer [CTUh]")
boxI.set_xlabel("Freshwater acidification [kg SO2-eq.]")
boxJ.set_xlabel("Terrestrial acidification [kg SO2-eq.]")
boxK.set_xlabel("Freshwater eutrophication [kg PO4 P-lim eq.]")
boxL.set_xlabel("Marine eutrophication [kg N N-lim eq.]")
boxM.set_xlabel("Land transformation, biodiversity [m2 arable land eq.]")
boxN.set_xlabel("Land occupation, biodiversity [m2 arable land eq. · yr]")
boxO.set_xlabel("Particulate matter formation [kg PM2.5-eq.]")
boxP.set_xlabel("Ionizing radiations [Bq C-14 eq.]")

plt.savefig('optimize_Ionizing radiations.png')

