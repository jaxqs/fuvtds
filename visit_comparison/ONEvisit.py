from data_models import get_new_data
from ONEfunc import plot_flux, plot_net
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    # Change these parameters to what is specific to you
    PID = '17249'
    visit = '12'

    data = get_new_data(PID, visit)

    pdf = PdfPages(f'output/{PID}_visit{visit}_flux.pdf')
    for d in data:
        plot_flux(d)
        pdf.savefig()
    pdf.close()

    pdf = PdfPages(f'output/{PID}_visit{visit}_net.pdf')
    for d in data:
        plot_net(d)
        pdf.savefig()
    pdf.close()