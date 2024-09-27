# Indonesian EEIO Model
# Sept 2024
# Ref: Putra, A. S., & Anita, Y. (2024). How does the establishment of an Indonesian Environmentally Extended Input Output (EEIO) model pave the way for Indonesia’s carbon future? Energy, Ecology and Environment. https://doi.org/10.1007/s40974-024-00328-6
# Python version: 3.10.11


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st

from util_app import save_toGit_csv


def calc_matrix_A (X_io, X_total):
    #print("inside calc_matrix_A")

    mat_A=np.divide(X_io,X_total)
    
    #print(mat_A.shape, X_total.shape)
    #print(mat_A[:5,:5])
    #print(X_total)

    return mat_A

def get_mat_finEnerCons(mat_A, X_FEC_yr):
    #FEC: final energy consumption

    #https://stackoverflow.com/questions/71443487/how-to-extract-non-consecutive-rows-and-columns-of-a-matrix

    rows=np.array([36, 37, 38, 144])
    #print("rows", rows)
    X1=mat_A[rows,:]

    total_1_axis = np.sum(X1, axis=1)
    #print(f'Sum of elements at 1-axis is {total_1_axis}')

    #https://stackoverflow.com/questions/35661919/numpy-array-divide-column-by-vector
    divider=total_1_axis.reshape(-1,1) 
    X1_div=np.divide(X1, divider)

    mult=X_FEC_yr.reshape(-1,1)
    #print("X1_div,mult shape", X1_div.shape, mult.shape, X1_div[:, :5], mult)
    X1_mult=np.multiply(X1_div,mult)
    #print("mult", mult)

    #print ("X1", X1[:,:5])
    #print ("X1_div", X1_div[:,:5])
    #print ("X1_mult", X1_mult[:,:5])
    return X1_mult

def get_mat_finCons(finEnerCons, convFactor):
    div1=convFactor.reshape(-1,1)
    X1_div=np.divide(finEnerCons,div1)
    X1_div=np.multiply(X1_div,1000)
    #print("div1", div1)

    #print ("X1", X1[:,:5])
    #print ("X1_div", X1_div[:,:5])
    #print ("X1_mult", X1_mult[:,:5])
    return X1_div

def get_mat_finConsCO2(finCons, CO2_EF):

    #print("CO2_EF", CO2_EF, CO2_EF.shape)

    #coal
    #=(B7*1000)*'Direct CO2 emission factor'!$B$9*'Direct CO2 emission factor'!$C$9
    row1=np.multiply(finCons[0,:],1000)
    row1=np.multiply(row1,CO2_EF[0,0])
    row1=np.multiply(row1,CO2_EF[0,1])

    #fuel
    #=B8*'Direct CO2 emission factor'!$B$16*'Direct CO2 emission factor'!$C$16
    row2=np.multiply(finCons[1,:],CO2_EF[1,0])
    row2=np.multiply(row2,CO2_EF[1,1])

    #nat gas
    #=B9*26.8*('Direct CO2 emission factor'!$B$14/1000)*'Direct CO2 emission factor'!$C$14
    row3=np.multiply(finCons[2,:],26.8)
    row3=np.multiply(row3,CO2_EF[2,0])
    row3=np.divide(row3,1000)
    row3=np.multiply(row3,CO2_EF[2,1])

    #electricity
    #=(B10*'Direct CO2 emission factor'!$B$24)*1000000
    row4=np.multiply(finCons[3,:],CO2_EF[3,0])
    row4=np.multiply(row4,1000000)

    mat_co2=np.vstack((row1,row2,row3,row4))

    #print("shape", mat_co2.shape)

    #print ("mat_co2", mat_co2[:,:5])

    return mat_co2

def get_io_aggregate(df_io, df_agg_label, totEm, scope1, scope2, scope3):
    #https://www.geeksforgeeks.org/pandas-groupby-and-sum/
    #https://stackoverflow.com/questions/37697195/how-to-merge-two-data-frames-based-on-particular-column-in-pandas-python
    #https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

    cnf_sector=df_agg_label.shape[0]
    #print("cnf_sector", cnf_sector)

    #df_agg_io=df_io[["Sector_Code",	"Sector_Name", "Sector_CodeName"]]
    df_agg_io=df_io[["Sector_CodeName"]]
    df_agg_io=df_agg_io.head(cnf_sector) #get the first 185 sectors

    df_agg_io["total_Emission"] = totEm
    df_agg_io["scope1_Emission"] = scope1
    df_agg_io["scope2_Emission"] = scope2
    df_agg_io["scope3_Emission"] = scope3

    df_agg_io=pd.merge(df_agg_io, df_agg_label, on='Sector_CodeName')
    #print(df_agg_io.head(5))

    df_agg_io=df_agg_io.drop(columns=['Sector_CodeName', 'Sector_Number'])
    #print(df_agg_io.head(5))

    #df1 = df_agg_io.groupby("Aggregated sectors")["total_Emission", "scope1_Emission", "scope2_Emission", "scope3_Emission"].sum()
    df_agg_sectors= df_agg_io.groupby("Aggregated sectors").sum()
    #print(df_agg_sectors.head(5))
    
    return df_agg_sectors

def get_aggregate_each ():
    df_emission = pd.read_csv("buf/result_emission.csv")
    df_io = pd.read_csv("data/io_ind_2016.csv")

    cnf_sector=df_emission.shape[0]
    df_sector=df_io[["Sector_CodeName"]]
    df_sector=df_sector.head(cnf_sector)

    df_emission["Sector_CodeName"]=df_sector.values.ravel()

    file_path_AGG = 'data/aggregated_sectors.csv'
    df_agg_label= pd.read_csv(file_path_AGG)

    df_emission=pd.merge(df_emission, df_agg_label, on='Sector_CodeName')

    return df_emission

def plot_agg_sectors(df):
    #https://dataviz.unhcr.org/tools/python/python_stacked_bar_chart.html
    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from textwrap import wrap

    #plt.style.use(['unhcrpyplotstyle','bar'])
    print (df.head(5))
    #sort by total descending order
    #df['Total'] = df[["scope1_Emission", "scope2_Emission", "scope3_Emission"]].sum(numeric_only=True, axis=1)
    df = df.sort_values("total_Emission", ascending=True)
    #print(df)
    
    #prepare data array for plotting
    x = df.index.values
    #print("x", x)
    
    y1 = df[['scope1_Emission']].values.ravel()
    y2 = df[['scope2_Emission']].values.ravel()
    y3 = df[['scope3_Emission']].values.ravel()
    #print("y1", y1)
    #b_y3 = np.add(y1, y2)

    #wrap long labels
    x = [ '\n'.join(wrap(l, 25)) for l in x ]
    
    #plot the chart
    fig, ax = plt.subplots()
    rect1=ax.barh(x, y1, label='Scope 1 Emission')
    rect2=ax.barh(x, y2, left=y1, label='Scope 2 Emission')
    rect3=ax.barh(x, y3, left=y1+y2, label='Scope 3 Emission')

    #set chart title
    ax.set_title('Scope 1, 2 and 3 emission 2016 (t CO2)')

    #set chart legend
    #ax.legend(loc=(0,1.02), ncol=3)
    ax.legend(loc="lower right", prop={'size': 10})

    #set y-axis title
    ax.set_xlabel('in t CO2', fontsize=12)
    ax.set_ylabel('Aggregated I/O sectors', fontsize=12)
    ax.tick_params(labelsize=9)
    
    #set y-axis label 
    ax.tick_params(labelbottom=True)

    #show grid below the bars
    ax.grid(axis='x')

    #format x-axis tick labels
    def number_formatter(x, pos):
        if x >= 1e6:
            s = '{:1.0f}M'.format(x*1e-6)
        elif x < 1e6 and x >= 1e3:
            s = '{:1.0f}K'.format(x*1e-3)
        else: 
            s = '{:1.0f}'.format(x)
        return s
    ax.xaxis.set_major_formatter(number_formatter)

    #set chart source and copyright
    #plt.annotate('Source: UNHCR Refugee Data Finder', (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = '#666666', fontsize=9)
    #plt.annotate('©UNHCR, The UN Refugee Agency', (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', color = '#666666', fontsize=9)

    #adjust chart margin and layout
    fig.tight_layout()

    #show chart
    plt.show()
    

def calc_mat(in_fname_io, in_fname_fec, in_fname_conv, in_fname_co2):
    print ("Inside eeio::calc_mat")
    print ("in_fname", in_fname_io, in_fname_fec, in_fname_conv, in_fname_co2)
    #Original IO table
    #-----------------------------------------------------------------------------------------
    #file_path = 'data/io_ind_2016.xlsx'
    file_path=f"data/{in_fname_io}"

    #pip install pandas openpyxl
    #df_io = pd.read_excel(file_path)
    df_io = pd.read_csv(file_path)
    #print(df_io.head(5))

    CNT_SECTOR=185
    sector_codeName=df_io.columns.values[3:CNT_SECTOR+3]
    #print(sector_codeName.shape, '\n',sector_codeName[:5])
    X = df_io[sector_codeName].values
    X_io=X[:CNT_SECTOR,:]
    X_total=X[-1,:]
    #print(X_io.shape, X_total.shape)
    #print(X_io[:5,:5]); print(X_total[:5])

    #Matrix A Coefficient
    #-----------------------------------------------------------------------------------------
    mat_A=calc_matrix_A(X_io, X_total)
    #print(mat_A[:5,:5])

    #Energy use per sector
    #-----------------------------------------------------------------------------------------
    #file_path_FEC = 'data/final_energy_consumption_bytype.xlsx'
    file_path_FEC = f"data/{in_fname_fec}"
    df_fec= pd.read_csv(file_path_FEC)
    #print(df_fec.columns.values)
    X_fec = df_fec[["Coal", "Fuel", "Natural gas", "Electricity"]].values
    ROW_Year=6 #row index for year 2016 inside the file
    X_fec_yr = X_fec[ROW_Year,:]
    mat_finEnerCons=get_mat_finEnerCons(mat_A, X_fec_yr)
    #pd.DataFrame(mat_finEnerCons).to_excel("buf/mat_finEnerCons.xlsx")

    #file_path_conv = 'data/conversion_factor.xlsx'
    file_path_conv = f"data/{in_fname_conv}"
    df_conv= pd.read_csv(file_path_conv)
    #print(df_conv.columns.values)
    X_conv = df_conv[["Multiplier Factor to BOE"]].values
    #print(X_conv, X_conv.shape)
    COL=[3, 31, 17, 38] #col index for Coal, Fuel, Nat Gas, Electricity
    X_conv_subset = X_conv[COL]
    mat_finCons=get_mat_finCons(mat_finEnerCons, X_conv_subset)
    #pd.DataFrame(mat_finCons).to_excel("buf/mat_finCons.xlsx")

    #file_path_co2 = 'data/direct_CO2_EF.xlsx'
    file_path_co2 = f"data/{in_fname_co2}"
    df_co2= pd.read_csv(file_path_co2)
    #print(df_co2.columns.values)
    X_co2 = df_co2[["Heat content (HHV)", "Emission Factor"]].values
    #print(X_co2, X_co2.shape)
    ROW=[0, 2, 1, 3] #col index for Coal, Fuel, Nat Gas, Electricity
    X_co2_subset = X_co2[ROW,:]
    mat_finConsCO2=get_mat_finConsCO2(mat_finCons, X_co2_subset)
    #pd.DataFrame(mat_finConsCO2).to_excel("buf/mat_finConsCO2.xlsx")

    total_finConsCO2_inG=np.sum(mat_finConsCO2, axis=0)
    #print(total_finConsCO2_inG[:5])
    total_finConsCO2_inT=np.divide(total_finConsCO2_inG,1000000)
    #print(total_finConsCO2_inT[:5])
    total_finConsCO2_inT_perMillion=np.divide(total_finConsCO2_inT,X_total)
    #print(total_finConsCO2_inT_perMillion[:5])

    total_finConsCO2_El_inT=np.divide(mat_finConsCO2[3,:],1000000) #electricity at the 4th row
    total_finConsCO2_El_inT_perMillion=np.divide(total_finConsCO2_El_inT, X_total)

    #Matrix B
    #-----------------------------------------------------------------------------------------
    mat_B = total_finConsCO2_inT_perMillion
    mat_B_El = total_finConsCO2_El_inT_perMillion

    #Matrix F & I
    #-----------------------------------------------------------------------------------------
    mat_F = np.identity(CNT_SECTOR)
    mat_I = np.identity(CNT_SECTOR)
    #print("mat_F", mat_F[:5,:5])

    #Calculation Scope 1
    #-----------------------------------------------------------------------------------------
    mat_BF = np.matmul(mat_B, mat_F)
    #print("mat_BF shape", mat_BF.shape); print("mat_BF", mat_BF[:5])

    #Calculation Total
    #-----------------------------------------------------------------------------------------
    mat_IminusA = np.subtract(mat_I, mat_A)
    #print("mat_IminusA  shape", mat_IminusA.shape); print("mat_IminusA ", mat_IminusA [:5,:5])

    mat_Inv=np.linalg.inv(mat_IminusA)
    #print("mat_Inv shape", mat_Inv.shape); print("mat_Inv ", mat_Inv [:5,:5])

    mat_InvF=np.matmul(mat_Inv, mat_F)
    #print("mat_InvF shape", mat_InvF.shape); print("mat_InvF", mat_InvF[:5,:5])

    mat_BInvF=np.matmul(mat_B, mat_InvF)
    #print("mat_BInvF shape", mat_BInvF.shape); print("mat_BInvF", mat_BInvF[:5])


    #Calculation Scope 2
    #-----------------------------------------------------------------------------------------
    mat_AF = np.matmul(mat_A, mat_F)
    #print("mat_AF shape", mat_AF.shape); print("mat_AF", mat_AF[:5, :5])

    mat_BAF = np.matmul(mat_B_El, mat_AF)
    #print("mat_BAF shape", mat_BAF.shape); print("mat_BAF", mat_BAF[:5])

    #Results
    #-----------------------------------------------------------------------------------------
    totalEmissionIntensity = mat_BInvF
    emissionIntensityScope1 = mat_BF
    emissionIntensityScope2 = mat_BAF
    emissionIntensityScope3 = np.subtract(mat_BInvF, np.add(mat_BF,mat_BAF))
    #print("emissionIntensityScope3 shape", emissionIntensityScope3.shape); print(emissionIntensityScope3[:5])

    df_emissionIntensity = pd.DataFrame(data=[emissionIntensityScope1, emissionIntensityScope2, emissionIntensityScope3, totalEmissionIntensity]).T
    df_emissionIntensity.columns = ["emissionIntensityScope1", "emissionIntensityScope2", "emissionIntensityScope3", "totalEmissionIntensity"]

    X_domOut = df_io[["7000/Total Domestic Output at Basic Price"]].values
    X_domOut=X_domOut[:CNT_SECTOR].ravel()
    totalEmission = totalEmissionIntensity * X_domOut
    #print(totalEmissionIntensity.shape, X_domOut.shape)
    #print("totalEmission shape", totalEmission.shape); print(totalEmission[:5])

    scope1_emission_inT = emissionIntensityScope1 * X_domOut
    scope2_emission_inT = emissionIntensityScope2 * X_domOut
    scope3_emission_inT = totalEmission - (scope1_emission_inT + scope2_emission_inT)
    #print("scope3_emission_inT", totalEmission.shape); print(scope3_emission_inT[:5])

    df_emission = pd.DataFrame(data=[scope1_emission_inT, scope2_emission_inT, scope3_emission_inT, totalEmission]).T
    df_emission.columns = ["emissionScope1", "emissionScope2", "emissionScope3", "totalEmission"]


    file_path_AGG = 'data/aggregated_sectors.csv'
    #file_path_AGG = in_path_agg
    df_agg_label= pd.read_csv(file_path_AGG)
    #print(df_fec.columns.values)

    df_agg_sectors=get_io_aggregate(df_io, df_agg_label, totalEmission, scope1_emission_inT, scope2_emission_inT, scope3_emission_inT)
    #print(df_agg_sectors.head(5))

    df_agg_sectors_each=get_aggregate_each()

    if (0):
        print ("Writing dataframe to excels...")

        pd.DataFrame(mat_A).to_excel("buf/mat_A.xlsx")
        pd.DataFrame(mat_B).to_excel("buf/mat_B.xlsx")
        pd.DataFrame(mat_B_El).to_excel("buf/mat_B_El.xlsx")
        pd.DataFrame(mat_I).to_excel("buf/mat_I.xlsx")
        pd.DataFrame(mat_F).to_excel("buf/mat_F.xlsx")
        pd.DataFrame(mat_BF).to_excel("buf/mat_BF.xlsx")
        pd.DataFrame(mat_IminusA).to_excel("buf/mat_IminusA.xlsx")
        pd.DataFrame(mat_Inv).to_excel("buf/mat_Inv.xlsx")
        pd.DataFrame(mat_InvF).to_excel("buf/mat_InvF.xlsx")
        pd.DataFrame(mat_BInvF).to_excel("buf/mat_BInvF.xlsx")
        pd.DataFrame(mat_AF).to_excel("buf/mat_AF.xlsx")
        pd.DataFrame(mat_BAF).to_excel("buf/mat_BAF.xlsx")

        df_emissionIntensity.to_excel("buf/result_emissionIntensity.xlsx")
        df_emission.to_excel("buf/result_emission.xlsx")
        df_agg_sectors.to_excel("buf/result_agg_sectors.xlsx")

    #from _init_var import * ##contains st.session_state['IS_LOCAL']

    if (st.session_state['IS_LOCAL']):
        print ("Writing dataframe to csv to localhost...")

        pd.DataFrame(mat_A).to_csv("buf/mat_A.csv", index=False)
        pd.DataFrame(mat_B).to_csv("buf/mat_B.csv", index=False)
        pd.DataFrame(mat_B_El).to_csv("buf/mat_B_El.csv", index=False)
        pd.DataFrame(mat_I).to_csv("buf/mat_I.csv", index=False)
        pd.DataFrame(mat_F).to_csv("buf/mat_F.csv", index=False)
        pd.DataFrame(mat_BF).to_csv("buf/mat_BF.csv", index=False)
        pd.DataFrame(mat_IminusA).to_csv("buf/mat_IminusA.csv", index=False)
        pd.DataFrame(mat_Inv).to_csv("buf/mat_Inv.csv", index=False)
        pd.DataFrame(mat_InvF).to_csv("buf/mat_InvF.csv", index=False)
        pd.DataFrame(mat_BInvF).to_csv("buf/mat_BInvF.csv", index=False)
        pd.DataFrame(mat_AF).to_csv("buf/mat_AF.csv", index=False)
        pd.DataFrame(mat_BAF).to_csv("buf/mat_BAF.csv", index=False)

        df_emissionIntensity.to_csv("buf/result_emissionIntensity.csv", index=False)
        df_emission.to_csv("buf/result_emission.csv", index=False)
        df_agg_sectors.to_csv("buf/result_agg_sectors.csv", index=True)

        df_agg_sectors_each.to_csv("buf/result_agg_sectors_each.csv",index=False)
    else:
        print ("Writing dataframe to github...")
        folder_name="buf"
        save_toGit_csv(pd.DataFrame(mat_A), "mat_A.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_B), "mat_B.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_B_El), "mat_B_El.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_I), "mat_B_El.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_F), "mat_F.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_BF), "mat_BF.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_IminusA), "mat_IminusA.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_Inv), "mat_Inv.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_InvF), "mat_InvF.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_BInvF), "mat_BInvF.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_AF), "mat_AF.csv", folder_name)
        save_toGit_csv(pd.DataFrame(mat_BAF), "mat_BAF.csv", folder_name)

        save_toGit_csv(df_emissionIntensity, "result_emissionIntensity.csv", folder_name)
        save_toGit_csv(df_emission, "result_emission.csv", folder_name)
        save_toGit_csv(df_agg_sectors, "result_agg_sectors.csv", folder_name, True)

        save_toGit_csv(df_agg_sectors_each, "result_agg_sectors_each.csv", folder_name)

    return df_agg_sectors    

	
	
	
## MAIN ##------------------------------------------------------------------------------------    
if __name__ == "__main__":
    IS_CALC_MATRIX = 1   #0 read matrix; 1 calculate matrix
    if 'IS_LOCAL' not in st.session_state:
        st.session_state['IS_LOCAL'] = 1

    if (IS_CALC_MATRIX):
        file_name_io = 'io_ind_2016.csv'
        file_name_FEC = 'final_energy_consumption_bytype.csv'
        file_name_conv = 'conversion_factor.csv'
        file_name_co2 = 'direct_CO2_EF.csv'
        file_name_AGG = 'aggregated_sectors.csv'

        df_agg_sectors = calc_mat(file_name_io, file_name_FEC, file_name_conv, file_name_co2)

    else:
        print ("Reading saved file to dataframe...")

        saved_file_path = "buf/result_agg_sectors.csv"
        df_agg_sectors = pd.read_excel(saved_file_path)
        df_agg_sectors.set_index('Aggregated sectors', inplace=True)
    #endif

    plot_agg_sectors(df_agg_sectors)


