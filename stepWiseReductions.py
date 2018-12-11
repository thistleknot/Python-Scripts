#without prefix, the next stepwise command gets confused on column names...
#would be best to only pass x, and then backjoin to these other derived values to do step_wise on
set1 = pd.concat([x,xLagged.add_prefix('xLag_'),(x*xLagged).add_prefix('xInter_'),y.add_prefix('y_'),yYield.add_prefix('yYield_')], names=['symbols'],axis=1)


X_train, X_test, y_train, y_test = train_test_split(x.loc[2:,][:-1], yFutureYield.loc[2:,][:-1], test_size=0.25)

print(result)

#filtered
my_list = set1[result]

#all columns
all_list = list(set1)

list2 = result
for i in range(0, len(list(result))):
    
        xInter_ = "xInter_" + result[i]
        list2.append(xInter_)
        xLag_ = "xLag_" + result[i]
        list2.append(xLag_)
        
        continue
		
list3 = result2
for i in range(0, len(list(result2))):
    
        temp = result2[i]
        #tempXLag="xLag_" + result2[i]
        #tempXInter = "xInter_" + result2[i]
        
        #print(result2[i].startswith( 'xInter_' ))        
        if result2[i].startswith( 'xInter_' ):
            print(result2[i][7:])
            list3.append(result2[i][7:])           
            list3.append('xLag_' + result2[i][7:]) 
            
            continue
        
            #print(result2[i].startswith( 'xLag_' ))
        elif result2[i].startswith( 'xLag_' ):
            print(result2[i][5:])
            list3.append(result2[i][5:])
            list3.append('xInter_' + result2[i][5:])
            continue
            #continue
            
        else:
            list3.append('xInter_' + result2[i])
            list3.append('xLag_' + result2[i]) 
            continue
    
print(list3)

print(list2)
#https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index
set1[list2].loc[train_index]

result2 = stepwise_selection(set1[list2].loc[train_index], y_train)
print(result2)

#model_training = sm.OLS(y_train,X_train,missing = 'drop').fit()
#X_train.filter(items=[result], axis=0)

#x.filter(regex=result)
    
list(set(list3))