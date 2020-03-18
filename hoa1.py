# 10.008 Hands-on Activity 1 Plotting Script
# This will spawn application window and browser tab popups to display the graphs
# NOTE: Ignore the pandas.util.testing and plotly.graph_objs.Marker deprecation warnings in the console when running this script
# Created by James Raphael Tiovalen (2020)

# Setup plotly
import chart_studio
chart_studio.tools.set_credentials_file(username='', api_key='')
chart_studio.tools.set_config_file(world_readable=False, sharing='private')
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.io as pio

# Import scientific libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn import metrics
from sympy import symbols, diff, integrate, lambdify, exp
import matplotlib.pyplot as plt
import seaborn as sns

# Define exponential function to fit to
def exponential(x, a, b):
    return a * np.exp(- b * (x - 61.9))

# Define output power function
def output(rl, rin, vse):
    return ((vse ** 2) * rl) / ((rin + rl) ** 2)

# Plot exponential graphs
def plot_exponential_graph_set(time, t_c, t_h, power, resistance, room_temp):
    data_dict = {"T_c": (t_c, "#17becf", 40, 20), "T_h": (t_h, "#de1738", 40, -20), "\dot{W}": (power, "#7851a9", 40, -20)}

    scatters = []
    trendlines = []
    annotations = []

    for dataname, datapair in data_dict.items():
        # Use dogleg algorithm
        popt, pcov = curve_fit(exponential, time, datapair[0], method="dogbox")
        
        # Prep constants for return values
        if dataname == "T_c":
            tc_coeffs = popt
        elif dataname == "T_h":
            th_coeffs = popt
        elif dataname == "\dot{W}":
            w_coeffs = popt

        # Compute R^2 value
        r_squared = round(metrics.r2_score(exponential(time, *popt), datapair[0]), 3)

        xx = np.linspace(time[0], time[-1], 2000)
        yy = exponential(xx, *popt)
        
        if dataname == "T_c" or dataname == "T_h":
            # Creating the dataset, and generating the plot
            scatters.append(go.Marker(
                              x=time,
                              y=datapair[0] + room_temp,
                              mode='markers',
                              marker=go.Marker(color=list(range(140)), colorscale="Viridis"),
                              name="",
                              showlegend=False
                              ))

            trendlines.append(go.Marker(
                              x=xx,
                              y=yy + room_temp,
                              mode='lines',
                              marker=go.Marker(color=datapair[1]),
                              name='${{{}}}$'.format(dataname),
                              ))

            annotations.append(go.layout.Annotation(
                              x=time[-20],
                              y=exponential(time[-20], *popt) + room_temp,
                              text='${} = {}e^{{-{}(t - 61.9)}} + {{{}}}$'.format(dataname, round(popt[0], 3), round(popt[1], 3), room_temp),
                              showarrow=True,
                              ax=datapair[2],
                              ay=datapair[3],
                              ))

            annotations.append(go.layout.Annotation(
                              x=time[-1],
                              y=exponential(time[-1], *popt) + room_temp,
                              text='$R^2 = {}$'.format(r_squared),
                              showarrow=True,
                              ax=datapair[2],
                              ay=datapair[3],
                              ))

        elif dataname == "\dot{W}":
            scatters.append(go.Marker(
                              x=time,
                              y=datapair[0],
                              mode='markers',
                              marker=go.Marker(color=list(range(140)), colorscale="Viridis"),
                              name="",
                              showlegend=False
                              ))

            trendlines.append(go.Marker(
                              x=xx,
                              y=yy,
                              mode='lines',
                              marker=go.Marker(color=datapair[1]),
                              name='${{{}}}$'.format(dataname),
                              ))

            annotations.append(go.layout.Annotation(
                              x=time[-20],
                              y=exponential(time[-20], *popt),
                              text='${} = {}e^{{-{}(t - 61.9)}}$'.format(dataname, round(popt[0], 3), round(popt[1], 3)),
                              showarrow=True,
                              ax=datapair[2],
                              ay=datapair[3],
                              ))

            annotations.append(go.layout.Annotation(
                              x=time[-1],
                              y=exponential(time[-1], *popt),
                              text='$R^2 = {}$'.format(r_squared),
                              showarrow=True,
                              ax=datapair[2],
                              ay=datapair[3],
                              ))

    layout = go.Layout(
                title='$\\textbf{{Graph for }} {} \\textbf{{ }} \Omega$'.format(resistance),
                plot_bgcolor='rgb(229, 229, 229)',
                xaxis=go.layout.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$t (s)$'),
                yaxis=go.layout.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$T (^\circ C)$'),
                annotations=annotations,
                )

    scatter_trendline_pair = list(zip(scatters, trendlines))
    final_data = [item for sublist in scatter_trendline_pair for item in sublist]
    fig = go.Figure(data=final_data, layout=layout)

    # Show plot on localhost
    pio.show(fig)
    # pio.write_html(fig, file='T_c vs Time', auto_open=True)
    return tc_coeffs, th_coeffs, w_coeffs

def plot_qc_qh_and_w(time, tc_coeffs, th_coeffs, resistance, room_temp):
    # Define symbols for differentiation
    t, Tc, Th = symbols("t Tc Th")
    Tc = tc_coeffs[0] * exp(-tc_coeffs[1] * (t - 61.9)) + room_temp
    Th = th_coeffs[0] * exp(-th_coeffs[1] * (t - 61.9)) + room_temp

    dTcdt = diff(Tc, t)
    dThdt = diff(Th, t)

    # mc = 21.87 J/°C
    dQcdt = 21.87 * dTcdt
    dQhdt = 21.87 * dThdt
    dWdt = dQhdt + dQcdt
    actual_Qc = lambdify(t, dQcdt)
    actual_Qh = lambdify(t, dQhdt)
    actual_W = lambdify(t, dWdt)

    # Start plotting
    data_dict = {"\dot{Q_c}": (tc_coeffs[1], actual_Qc, "#17becf", dQcdt, -40, 20), "\dot{Q_h}": (th_coeffs[1], actual_Qh, "#de1738", dQhdt, 40, -20), "\dot{W}": (th_coeffs[1] + tc_coeffs[1], actual_W, "#7851a9", dWdt, 40, 20)}
    plots = []
    annotations = []

    for dataname, datapair in data_dict.items():
        xx = np.linspace(time[0], time[-1], 2000)
        yy = abs(datapair[1](xx))

        plots.append(go.Marker(
                          x=xx,
                          y=yy,
                          mode='lines',
                          marker=go.Marker(color=datapair[2]),
                          name='${{{}}}$'.format(dataname),
                          ))

        if dataname == "\dot{Q_c}" or dataname == "\dot{Q_h}":
            annotations.append(go.layout.Annotation(
                          x=time[20],
                          y=abs(datapair[1](time[20])),
                          text='${} = {}e^{{-{}t}}$'.format(dataname, abs(round(datapair[3].args[0], 3)), round(datapair[0], 3)),
                          showarrow=True,
                          ax=datapair[4],
                          ay=datapair[5],
                          ))

        elif dataname == "\dot{W}":
            annotations.append(go.layout.Annotation(
                          x=time[-20],
                          y=datapair[1](time[-20]),
                          text='${} = {}e^{{-{}t}} - {}e^{{-{}t}}$'.format(dataname, abs(round(dQhdt.args[0], 3)), round(th_coeffs[1], 3), abs(round(dQcdt.args[0], 3)), round(tc_coeffs[1], 3)),
                          showarrow=True,
                          ax=datapair[4],
                          ay=datapair[5],
                          ))

    layout = go.Layout(
                title='$\\textbf{{Graph for }} {} \\textbf{{ }} \Omega$'.format(resistance),
                plot_bgcolor='rgb(229, 229, 229)',
                xaxis=go.layout.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$t (s)$'),
                yaxis=go.layout.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$\dot{Q} (J/s)$'),
                annotations=annotations,
                )

    fig = go.Figure(data=plots, layout=layout)

    # Show plot on localhost
    pio.show(fig)

# Trapezoidal rule method
def stage_1_sum(stage_1_time, stage_1_power, resistance):
    total_energy = 0
    for i in range(len(stage_1_time) - 1):
        total_energy += ((stage_1_power[i] + stage_1_power[i+1]) / 2) * (stage_1_time[i+1] - stage_1_time[i])
    
    return total_energy

# Integration method
def stage_2_sum(time, w_coeffs, resistance):
    t, W = symbols("t W")
    W = w_coeffs[0] * exp(-w_coeffs[1] * (t - 61.9))

    IWdt = integrate(W, (t, time[0], time[-1]))
    
    return IWdt

# Plot boxplot to detect anomalies
def detect_anomalies(output_power_values):
    sns.set(style="whitegrid")
    ax = sns.boxplot(x=output_power_values, orient="h", palette="Set3", flierprops=dict(markerfacecolor='r', markersize=5, linestyle='none'))
    ax.set(xlabel="Output power, $\dot{W}$")
    plt.show()

# Plot actual boxplot
def plot_actual_boxplot(actual_output_power):
    sns.set(style="whitegrid")
    ax = sns.boxplot(x=actual_output_power, orient="h", palette="Set3", flierprops=dict(markerfacecolor='r', markersize=5, linestyle='none'))
    ax.set(xlabel="Actual output power, $\dot{W}$")
    plt.show()

# Manual plotting
def output_power_graph(resistance_values, output_power_values):
    scatters = []
    trendlines = []
    annotations = []
    
    # Use dogleg algorithm
    popt, pcov = curve_fit(output, resistance_values, output_power_values, method="dogbox")

    # Compute R^2 value
    r_squared = round(metrics.r2_score(output(resistance_values, *popt), output_power_values), 3)

    xx = np.linspace(resistance_values[0], resistance_values[-1], 2000)
    yy = output(xx, *popt)

    scatters.append(go.Marker(
                      x=resistance_values,
                      y=output_power_values,
                      mode='markers',
                      marker=go.Marker(color=list(range(10)), colorscale="Viridis"),
                      name="",
                      showlegend=False
                      ))

    trendlines.append(go.Marker(
                      x=xx,
                      y=yy,
                      mode='lines',
                      marker=go.Marker(color="#ff007f"),
                      name='${{\dot{W}}}$',
                      ))

    annotations.append(go.layout.Annotation(
                      x=resistance_values[-4],
                      y=output(resistance_values[-4], *popt),
                      text='$\dot{{W}} = \\frac{{{}^2 R_L}}{{{{{} + R_L}}^2}}$'.format(round(popt[1], 3), round(popt[0], 3)),
                      showarrow=True,
                      ax=70,
                      ay=-5,
                      ))

    annotations.append(go.layout.Annotation(
                      x=resistance_values[-1],
                      y=output(resistance_values[-1], *popt),
                      text='$R^2 = {}$'.format(r_squared),
                      showarrow=True,
                      ax=70,
                      ay=-5,
                      ))

    layout = go.Layout(
                title='$\\textbf{Output Power Graph}$',
                plot_bgcolor='rgb(229, 229, 229)',
                xaxis=go.layout.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$R_L (\Omega)$'),
                yaxis=go.layout.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)', title='$\dot{W} (W)$'),
                annotations=annotations,
                )

    scatter_trendline_pair = list(zip(scatters, trendlines))
    final_data = [item for sublist in scatter_trendline_pair for item in sublist]
    fig = go.Figure(data=final_data, layout=layout)

    # Show plot on localhost
    pio.show(fig)

# Function to graph (W, T_c and T_h) vs time
def plot_data_graphs(resistance):
    # Read xlsx file as dataframe
    df = pd.read_excel("dataset.xlsx", sheet_name=f"{resistance}ohm", usecols = "A:D", na_values=" ", skiprows=range(0, 67)).dropna()
    
    stage_1 = pd.read_excel("dataset.xlsx", sheet_name=f"{resistance}ohm", usecols = "A:D", na_values=" ", skiprows=[0], nrows=64).dropna()
    # Voltage is 3.0 V
    stage_1_power = stage_1[" Current(A)"].to_numpy() * 3
    stage_1_time = stage_1["time(s)"].to_numpy()

    # Get room temperature based on average of last two data (to minimise systematic error)
    room_temp = (df["CH1(C)"].to_numpy()[-1] + df["CH2(C)"].to_numpy()[-1]) / 2

    # Parse columns as numpy.ndarray
    time = df["time(s)"].to_numpy()
    t_c = df["CH1(C)"].to_numpy() - room_temp
    t_h = df["CH2(C)"].to_numpy() - room_temp
    power = df[" Voltage(V)"].to_numpy() ** 2 / resistance

    time = time.astype(np.float64)
    t_c = t_c.astype(np.float64)
    t_h = t_h.astype(np.float64)
    power = power.astype(np.float64)

    # Answer question (4)
    tc_coeffs, th_coeffs, w_coeffs = plot_exponential_graph_set(time, t_c, t_h, power, resistance, room_temp)
    # Answer question (5)
    plot_qc_qh_and_w(time, tc_coeffs, th_coeffs, resistance, room_temp)
    # Answer question (7)
    input = stage_1_sum(stage_1_time, stage_1_power, resistance)
    print(f"Energy input: {input} J")
    # Answer question (8)
    output = stage_2_sum(time, w_coeffs, resistance)
    print(f"Energy output: {output} J")
    # Answer question (9)
    efficiency = output/input * 100
    print(f"Efficiency: {efficiency}%")

if __name__ == '__main__':
    for resistance_values in [1]:
        plot_data_graphs(resistance_values)
    
    resistance_values = np.array([1, 1.2, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6]).astype(np.float64)
    voltage_values = np.array([0.176193, 0.193620666666667, 0.247881, 0.3116985, 0.317917, 0.3412875, 0.361429, 0.384375, 0.40167875]).astype(np.float64)
    output_power_values = voltage_values ** 2 / resistance_values
    output_power_values = output_power_values.astype(np.float64)
    
    detect_anomalies(output_power_values)
    
    # Remove anomaly
    actual_resistances = np.delete(resistance_values, [3])
    actual_voltages = np.delete(voltage_values, [3])
    
    actual_output_power = actual_voltages ** 2 / actual_resistances
    actual_output_power = actual_output_power.astype(np.float64)
    
    plot_actual_boxplot(actual_output_power)
    
    output_power_graph(actual_resistances, actual_output_power)