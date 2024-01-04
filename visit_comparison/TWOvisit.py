from data_models import get_new_data
from TWOfunc import organize, plot_flux
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    # Change these parameters to what is specific to you
    PID = '17249'
    visit_new = '11'
    visit_ref = '09'

    #Grab the data from COSMO
    data_new = get_new_data(PID, visit_new)
    data_ref = get_new_data(PID, visit_ref)

    # organize the data by cenwave and transform into DataFrames
    data_new = organize(data_new)
    data_ref = organize(data_ref)

    pdf = PdfPages(f'output/{PID}_visit{visit_new}&visit{visit_ref}_comparison.pdf')
    for c in data_new.columns:
        plot_flux(data_new[c], data_ref[c], c)
        pdf.savefig()
    pdf.close()