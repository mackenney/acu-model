import numpy as np

"""
Plot=0; %1=plot, 0=no plot
10 Plotinterval=100; %determines the interval between interations to plot
11
12 %Settings for Single or Double Precision
13 chooseSingle=1; %1=Single precision, 0=Double factor=1;
14
15 %Defining the time step, spacial step and speed of sound constants
K=1/factor*1.25e-4; %this corresponds to a sampling frequency of 8000Hz h=1/factor*.1; %this is a spatial sampling of 10 cm
16 c=344; %speed of sound
17 rho=1.21; %density of air
18 fs=1/K;
19
20 %n Sets number of time steps
21 n=8000*factor;
22 T=0:n;
"""

Plot = 0  # ; #1=plot, 0=no plot
Plotinterval = 100  # determines the interval between interations to plot

# Settings for Single or Double Precision
chooseSingle = 1  # 1=Single precision, 0=Double
factor = 1
# Defining the time step, spacial step and speed of sound constants

K = 1 / factor * 1.25e-4  # this corresponds to a sampling frequency of 8000Hz
h = 1 / factor * .1  # this is a spatial sampling of 10 cm

c = 344  # speed of sound
rho = 1.21;  # density of air
fs = 1 / K
# n Sets number of time steps
n = 8000 * factor;
T = np.arange(n)

# This sets the size of the room assume a P matrix with MxNxO
#
M = 42 * factor  # 42width 1 to 42, thus 41 points always one bigger because there is not zero position, array starts at 1
N = 56 * factor  # 56depth
O = 24 * factor  # 24height

# This will add in a seperate room connected to the main room
sub_room_flag = 0
""" CUIDADO!! con linea que sigue.  """
M_sub = 50 * factor  # you are re-using a Ux from the main room so you don't need to account for 0 position being 1
N_sub = 28 * factor + 1
O_sub = O  # <----DO NOT CHANGE, I have not accounted for any other case

# This will be where we add a doorway
Rightside_doorway_flag = 0

# Activating the speakers you want to test
# 1=active and all other values = deactive
sourceon = 0
centreon = 0
lefton = 1
righton = 1
rlefton = 1
rrighton = 1
subwooferon = 0
# defining the time delay and a relative amplitude for the rear speakers
"""Tengo dudas con esta linea también."""
timedelay = 130 * factor
amp = -0.85
amp_RearLeft = amp
amp_RearRight = amp

# determins parameters for plotting with correct aspect ratios
if M > N:
    largest_dim = M
elif N > M:
    largest_dim = N
elif M == N:
    largest_dim = M

# pre=allocating arrays
P = np.zeros((M - 1, N - 1, O - 1))
Ux = np.zeros((M, N - 1, O - 1))
Uy = np.zeros((M - 1, N, O - 1))
Uz = np.zeros((M - 1, N - 1, O))

"""Linea para ahorrar memoria, no se si la necesitamos."""
# if chooseSingle == 1:  # changes double arrays to single
#     P = single(P);
#     Ux = single(Ux);
#     Uy = single(Uy);
#     Uz = single(Uz);


# pre-allocating speakers
Pold_source = 0
Pold_centre = 0
Pold_left = 0
Pold_right = 0
Pold_rleft = 0
Pold_rleft = 0  # quizás debería ser Pold_rright
Pold_subwoofer = 0

# pre-allocating the mic array
# data stored as single precision to save space EN PYTHON: dtype=float
Psource = np.zeros((n + 1, 1))
Pmic1 = np.zeros((n + 1, 1))
Pmic2 = np.zeros((n + 1, 1))
Pmic3 = np.zeros((n + 1, 1))
Pmicarray = np.zeros((5, 5, n + 1))

# this will be for adding in the initial pulse
pulsewidth = 80 * factor  # in samples
offset = 0
Q = (0.5 * (1 - np.cos(2 * np.pi * (T) / pulsewidth))) ** 2
dQ = (np.pi / pulsewidth) * np.sin(2 * np.pi * (T) / pulsewidth) * (1 - np.cos(2 * np.pi * (T) / pulsewidth))

# if chooseSingle == 1:
#     Q = single(Q)


""" ChECK BOUNDARIES """
for i in range(pulsewidth + 1 + offset, n + 1):
    Q[i] = 0
    dQ[i + 1] = 0

# if chooseSingle ==1:
#     Q=single(Q);

Qfft = 1 / fs * np.fft.fft(Q, n);
# f = fs / n * (0:n / 2);
f = fs / n * (np.arange(n + 1) / 2)
""" Grafico """
# figure(1)
# Qplot = semilogx(f, 20 * log10(abs(Qfft(1:n / 2 + 1))));
# ylabel('[dB]');
# xlabel('Freq[Hz]');
# grid on axis([1 3000 - 100 - 30])
# set(Qplot, 'linewidth', 2)
# drawnow;


f = fs / n * (np.arange(n))
Qfft2 = f * Qfft;
""" Grafico """
# figure (2)
# Qplot=semilogx(f(1:n/2+1) ,20*log10(abs(Qfft2(1:n/2+1))));
# ylabel('[dB]');
# xlabel('Freq [Hz]');
# grid on axis([1 3000 -100 -30])
# set(Qplot,'linewidth',2)
# drawnow;

dQfft = 1 / fs * np.fft.fft(dQ, n);
f = fs / n * (np.arange(n + 1) / 2)
# figure (3)
# dQplot=semilogx(f,20*log10(abs(dQfft(1:n/2+1))));
# ylabel('[dB]');xlabel('Freq[Hz]');
# grid on axis ([1 3000 -100 -30])
# set(dQplot,'linewidth',2)
# drawnow;

# this is only active with the sub-room switch
if sub_room_flag == 1:
    P_sub = np.zeros((M_sub, N_sub - 1, O_sub - 1))  # (M_sub, N_sub - 1, O_sub - 1);
    Ux_sub = np.zeros((M_sub, N_sub - 1, O_sub - 1))
    Uy_sub = np.zeros((M_sub, N_sub, O_sub - 1))
    Uz_sub = np.zeros((M_sub, N_sub - 1, O_sub))
    Ux_subroom_old = Ux[M, 1:N_sub, :]  # original:Ux_subroom_old=Ux(M,1:N_sub -1,:);
    new_dimx = (M - 1 + M_sub);
    P_plot = 50 * np.ones((new_dimx, N - 1))
    # if chooseSingle == 1:  # changes double arrays to single
    #     P_sub = single(P_sub);
    #     Ux_sub = single(Ux_sub);
    #     Uy_sub = single(Uy_sub);
    #     Uz_sub = single(Uz_sub);

# DOORWAY

if Rightside_doorway_flag == 1:
    heightstart = 1  # <----DO NOT CHANGE, I have not accounted for any other case
    heightfinish = 20
    widthstart = 17
    widthfinish = 37
    Ux_doorway_old = Ux[M, widthstart:widthfinish + 1, heightstart:heightfinish + 1]

# determining the centre of the room for reference
roomcentrex = round((M) / 2)
roomcentrey = round((N) / 2)
roomcentrez = round((O) / 2)

# dummy source position in centre of the room
xsource = roomcentrex
ysource = roomcentrey
zsource = roomcentrez

# set Centre Channel position
xcentre = round(M / 2)
ycentre = N - 1
zcentre = round(O / 2)
move = 0

# set Right Channel Position
xright = (round((M) / 4)) - move
yright = N - 1
zright = round(O / 2)

# set Left Channel Position
xleft = M - xright + move
yleft = N - 1
zleft = round(O / 2)

# set Rear-Right Channel Position
xrright = xright
yrright = 1
zrright = round(O / 2)

# set Rear-Left Channel Position
xrleft = xleft
yrleft = 1
zrleft = round(O / 2)

# set Subwoofer Position
xsubwoofer = 1
ysubwoofer = 1
zsubwoofer = 1

# defining the absorption coefficients and wall impedances
alpha_right = .1
alpha_left = .1
alpha_front = .1
alpha_back = .1
alpha_roof = .1
alpha_floor = .1

Z_right = rho * c * ((1 + np.sqrt(1 - alpha_right)) / (1 - np.sqrt(1 - alpha_right)));
Z_left = rho * c * ((1 + np.sqrt(1 - alpha_left)) / (1 - np.sqrt(1 - alpha_left)));
Z_front = rho * c * ((1 + np.sqrt(1 - alpha_front)) / (1 - np.sqrt(1 - alpha_front)));
Z_back = rho * c * ((1 + np.sqrt(1 - alpha_back)) / (1 - np.sqrt(1 - alpha_back)));
Z_roof = rho * c * ((1 + np.sqrt(1 - alpha_roof)) / (1 - np.sqrt(1 - alpha_roof)));
Z_floor = rho * c * ((1 + np.sqrt(1 - alpha_floor)) / (1 - np.sqrt(1 - alpha_floor)));
Z_open_doorway = rho * c;

startloopscheckflag = 1
timestamp1 = 0
Pmicarray = np.zeros((5, 5, n + 1))  # allocating mic array

# Setting the constants

UxleftwallU = (rho * h - K * Z_left) / (rho * h + K * Z_left)
UxleftwallP = 2 * K / (rho * h + K * Z_left)
UxrightwallU = (rho * h - K * Z_right) / (rho * h + K * Z_right)
UxrightwallP = 2 * K / (rho * h + K * Z_right)
UxdoorwayU = (rho * h - K * Z_open_doorway) / (rho * h + K * Z_open_doorway)
UxdoorwayP = 2 * K / (rho * h + K * Z_open_doorway)
UybackwallU = (rho * h - K * Z_back) / (rho * h + K * Z_back)
UybackwallP = 2 * K / (rho * h + K * Z_back)
UyfrontwallU = (rho * h - K * Z_front) / (rho * h + K * Z_front)
UyfrontwallP = 2 * K / (rho * h + K * Z_front)
UzfloorU = (rho * h - K * Z_floor) / (rho * h + K * Z_floor)
UzfloorP = 2 * K / (rho * h + K * Z_floor)
UzroofU = (rho * h - K * Z_roof) / (rho * h + K * Z_roof)
UzroofP = 2 * K / (rho * h + K * Z_roof)

# for T=0:n;
for T in range(n + 1):
    # begin velocity iterations
    # side walls
    # left hand wall
    Ux[1, :, :] = UxleftwallU * Ux[1, :, :] - UxleftwallP * P[1, :, :];  # right hand wall
    Ux[M, :, :] = UxrightwallU * Ux[M, :, :] + UxrightwallP * P[M - 1, :, :]
    if Rightside_doorway_flag == 1:
        Ux[M, widthstart: widthfinish, heightstart: heightfinish] = \
            UxdoorwayU * Ux_doorway_old + UxdoorwayP * P[M - 1, widthstart: widthfinish, heightstart: heightfinish]
        Ux_doorway_old = Ux[M, widthstart:widthfinish, heightstart: heightfinish]
    elif sub_room_flag == 1:
        Ux[M, 1: N_sub - 1, :] = Ux_subroom_old + K / rho / h * (P[M - 1, 1: N_sub - 1, :] - P_sub[1, 1: N_sub - 1, :])
        Ux_subroom_old = Ux[M, 1:N_sub - 1, :]

    """ diff operator: difference between adjacent elements in array. diff(X,N, dim). N: order, dim: axis
        CHECK BOUNDARIES!!
    """
    # the rest
    Ux[2:M - 1, :, :] = Ux[2:M - 1, :, :] + K / rho / h * (-diff(P[2 - 1:M - 1, :, :], 1, 1))
    # front and back walls #left hand wall
    Uy[:, 1, :] = UybackwallU * Uy[:, 1, :] - UybackwallP * P[:, 1, :]
    # right hand wall
    Uy[:, N, :] = UyfrontwallU * Uy[:, N, :] + UyfrontwallP * P[:, N - 1, :]
    # the rest
    Uy[:, 2: N - 1, :] = Uy[:, 2: N - 1, :] + K / rho / h * (-diff(P[:, 1:N - 1, :], 1, 2))
    # roof and floor #left hand wall
    Uz[:, :, 1] = UzfloorU * Uz[:, :, 1] - UzfloorP * P[:, :, 1]
    # right hand wall
    Uz[:, :, O] = UzroofU * Uz[:, :, O] + UzroofP * P[:, :, O - 1]
    # the rest
    Uz[:, :, 2: O - 1] = Uz[:, :, 2: O - 1] + K / rho / h * (-diff(P[:, :, 1:O - 1], 1, 3))

    # now do the Pressure Matrix
    P = P + c * c * rho * K / h * (-diff(Ux, 1, 1) - diff(Uy, 1, 2) - diff(Uz, 1, 3));

    # this section adds in the speakers into the system with our defined Q
    if sourceon == 1:
        P[xsource, ysource, zsource] = \
            K * c * c * rho / (h * h * h) * Q[T + 1] + Pold_source + c * c * rho * K / h * (
                    Ux[xsource, ysource, zsource] - Ux[xsource + 1, ysource, zsource] +
                    Uy[xsource, ysource, zsource] - Uy[xsource, ysource + 1, zsource] +
                    Uz[xsource, ysource, zsource] - Uz[xsource, ysource, zsource + 1])

    if centreon == 1: 
        P[xcentre, ycentre, zcentre] =\
            K * c * c * rho / (h * h * h) * Q[T+1]+Pold_centre+c * c * rho * K / h * \
            (Ux[xcentre, ycentre, zcentre]- Ux[xcentre+1, ycentre, zcentre]+Uy[xcentre, ycentre, zcentre]-
             Uy[ xcentre, ycentre+1, zcentre]+Uz[xcentre, ycentre, zcentre]-Uz[xcentre, ycentre, zcentre+1]);

    if lefton == 1:
        P[xleft, yleft, zleft] =\
            K * c * c * rho / (h * h * h) * Q[T+1]+Pold_left+c * c * rho * K / h * \
            (Ux[xleft, yleft, zleft]-Ux[xleft+1, yleft, zleft]+Uy[xleft, yleft, zleft]-Uy[xleft, yleft+1, zleft]+
             Uz[xleft, yleft, zleft]-Uz[xleft, yleft, zleft+1]);

    if righton == 1:
        P[xright, yright, zright] =\
            K * c * c * rho / (h * h * h) * Q[T + 1] + Pold_right + c * c * rho * K / h * \
            (Ux[xright, yright, zright] - Ux[xright + 1, yright, zright] + Uy[xright, yright, zright] - 
             Uy[xright,yright + 1,zright] + Uz[xright, yright, zright] - Uz[xright, yright, zright + 1])

    if rlefton == 1 and T >= timedelay:
        P[xrleft, yrleft, zrleft] = \
            K * c * c * rho / (h * h * h) * amp_RearLeft * Q[T + 1 - timedelay] + Pold_rleft + c * c * rho * K / h * \
            (Ux[xrleft, yrleft, zrleft] - Ux[xrleft + 1, yrleft, zrleft] + Uy[xrleft, yrleft, zrleft] -
             Uy[xrleft, yrleft + 1, zrleft] + Uz[xrleft, yrleft, zrleft] - Uz[xrleft, yrleft, zrleft + 1])

    if rrighton == 1 and T >= timedelay:
        P[xrright, yrright, zrright] = \
            K * c * c * rho / (h * h * h) * amp_RearRight * Q[T + 1 - timedelay] + Pold_rright + c * c * rho * K / h * \
            (Ux[xrright, yrright, zrright] - Ux[xrright + 1, yrright, zrright] + Uy[xrright, yrright, zrright] -
             Uy[xrright, yrright + 1, zrright] + Uz[xrright, yrright, zrright] - Uz[xrright, yrright, zrright + 1])

    if subwooferon == 1:
        P[xsubwoofer, ysubwoofer, zsubwoofer] = \
            K * c * c * rho / (h * h * h) * Q[T + 1] + Pold_subwoofer + c * c * rho * K / h * \
            (Ux[xsubwoofer, ysubwoofer, zsubwoofer] - Ux[xsubwoofer + 1, ysubwoofer, zsubwoofer] +
             Uy[xsubwoofer, ysubwoofer, zsubwoofer] - Uy[xsubwoofer, ysubwoofer + 1, zsubwoofer] +
             Uz[xsubwoofer, ysubwoofer, zsubwoofer] - Uz[xsubwoofer, ysubwoofer, zsubwoofer + 1])

    # grabs and stores old source term data
    Pold_source = P(xsource, ysource, zsource);

    Pold_centre = P(xcentre, ycentre, zcentre);
    Pold_left = P(xleft, yleft, zleft);
    Pold_right = P(xright, yright, zright);
    Pold_rleft = P(xrleft, yrleft, zrleft);
    Pold_rright = P(xrright, yrright, zrright);

    Pold_subwoofer = P(xsubwoofer, ysubwoofer, zsubwoofer);

    if sub_room_flag == 1:
        Ux_sub[M_sub, :, :] = (rho * h - K * Z_right) / (rho * h + K * Z_right) * Ux_sub[M_sub, :, :] + 2 * K / \
                              (rho * h + K * Z_right) * P_sub[M_sub, :, :]
        Ux_sub[1: M_sub - 1, :, :] = Ux_sub[1: M_sub - 1, :, :] + K / rho / h * (-diff(P_sub[1:M_sub, :, :], 1, 1))

        # front and back walls
        #  left hand wall
        Uy_sub[:, 1, :] = (rho * h - K * Z_back) / (rho * h + K * Z_back) * Uy_sub[:, 1, :] - 2 * K / \
                          (rho * h + K * Z_back) * P_sub[:, 1, :];
        # right hand wall
        Uy_sub[:, N_sub, :] = (rho * h - K * Z_front) / (rho * h + K * Z_front) * Uy_sub[:, N_sub, :] + 2 * K / \
                              (rho * h + K * Z_front) * P_sub[:, N_sub - 1, :]
        # the rest
        Uy_sub[:, 2: N_sub - 1, :] = Uy_sub[:, 2: N_sub - 1, :] + K / rho / h * (-diff(P_sub[:, 1:N_sub - 1, :], 1, 2));

        # roof and floor
        # left hand wall
        Uz_sub[:, :, 1] = (rho * h - K * Z_floor) / (rho * h + K * Z_floor) * Uz_sub[:, :, 1] - 2 * K / (
                rho * h + K * Z_floor) * P_sub[:, :, 1];
        # right hand wall
        Uz_sub[:, :, O_sub] = (rho * h - K * Z_roof) / (rho * h + K * Z_roof) * Uz_sub[:, :, O_sub] + 2 * K / (
                rho * h + K * Z_roof) * P_sub[:, :, O_sub - 1];
        # the rest
        Uz_sub[:, :, 2: O_sub - 1] = Uz_sub[:, :, 2: O_sub - 1] + K / rho / h * (-diff(P_sub[:, :, 1:O_sub - 1], 1, 3));

        P_sub[1, :, :] = P_sub[1, :, :] + c * c * rho * K / h * (Ux[M, 1:N_sub - 1, 1:O_sub - 1] - Ux_sub[1, :, :] - \
                                                                 diff(Uy_sub[1, :, :], 1, 2) - diff(Uz_sub[1, :, :], 1,
                                                                                                    3));

        P_sub[2: M_sub, :, :] = P_sub[2: M_sub, :, :] + c * c * rho * K / h * (
                -diff(Ux_sub, 1, 1) - diff(Uy_sub[2:M_sub, :, :], 1, 2) - diff(Uz_sub[2: M_sub, :, :], 1, 3));

    # collecting mic  data
    # Source is the geometrical centre of the room

    Psource[T + 1] = P[xsource, ysource, zsource]
    Pmic1[T + 1] = P[15 * factor, 15 * factor, zsource]

    Pmic2[T + 1] = P[15 * factor, 20 * factor, zsource]
    Pmic3[T + 1] = P[20 * factor, 15 * factor, zsource]
    Pmic4[T + 1] = P[20 * factor, 20 * factor, zsource]

    # micarray
    a = 5 * factor;
    # width spacing in h cm increments
    b = 5 * factor;  # depth spacing
    # defining the mic array based on the approximate centre of the room

    for i in range(-2, 3):
        for j in range(-2, 3): \
                Pmicarray[3 + i, 3 + j, T + 1] = P[roomcentrex - i * a, roomcentrey - j * b, roomcentrez];

    if sub_room_flag == 1:
        P_plot(1: M - 1, 1: N - 1)=P(:,:, zsource);
        P_plot(M: M - 1 + M_sub, 1: N_sub - 1)=P_sub(:,:, zsource);
    else
        P_plot = P(:,:, zsource);

    """MATLAB TIME"""
    timestamp2 = toc;
    howmuchdone = T / n * 100
    timeperiteration = (timestamp2 - timestamp1) / Plotinterval
    timestamp1 = timestamp2

    if Plot == 1:
        if chooseSingle == 1:
            P_plot = double(P_plot);
        #
        # figure(4)
        # surf(P_plot);
        # title('pressure');
        # ylabel('x-axis');
        # xlabel('y-axis');
        # zlabel('amplitude')
        #
        # axis([0 largest_dim 0 largest_dim - 50 150]);
        # drawnow;



    357
    end
    358
    end

    # plotting mic and source responses
    for T in range(n+1):
        t(T + 1) = (T * K);

    figure(5)
    plot(t, Psource);
    title('Pressure at the Centre Point');
    ylabel('pressure amplitude');
    xlabel('time (s)');
    # axis([0.7 170]) grid on

    figure (6)
    370 plot(t,Pmic1);
    371 title('Pressure at the 1st mic');
    372 ylabel('pressure amplitude');xlabel('time (s)');
    373
    374 figure (7)
    375 plot(t,Pmic2);
    376 title('Pressure at the 2nd Mic');
    377 ylabel('pressure amplitude');xlabel('time (s)');
    378
    379 figure (8)
    380 plot(t,Pmic3);
    381 title('Pressure at the 3rd Mic');
    382 ylabel('pressure amplitude');xlabel('time (s)');
    383
    384 figure (9)
    385 plot(t,Pmic4);
    386 title('Pressure at the 4th Mic');
    387 ylabel('pressure amplitude');xlabel('time (s)');
    388
    389 Pmic1fft=1/fs*fft(Pmic1.*Pmic1,n);
    390 figure (10)
    391 Pmic1plot=plot(f,(abs(Pmic1fft(1:n/2+1))));
    392 ylabel('[dB]');xlabel('Freq [Hz]');
    393 grid on
    394 axis ([0 50 0 1200])
    395 set(Pmic1plot ,'linewidth',2)
    396
    397 Pmic1fft=1/fs*fft(Pmic1,n);
    398 figure (100)
    399 Pmic1plot=plot(f,(20*log10(Pmic1fft(1:n/2+1))));
    400 %title('Frequency Response in Groh Room from a sub-woofer in the
    corner with a 10msec pulse','FontSize',20)
    401 ylabel('Pressure [dB]','FontSize',16);xlabel('Frequency [Hz]','
    FontSize',16);
    402 grid on
    403 axis ([0 200 -50 10])
    404 legend('Un-Smoothed'); set(Pmic1plot,'linewidth',3)
    405
    406 Pmic2fft=1/fs*fft(Pmic2,n);
    407 figure (200)
    408 Pmic2plot=plot(f,(20*log10(Pmic2fft(1:n/2+1))));

409 %title('Frequency Response in Groh Room from a sub-woofer in the corner with a 10msec pulse','FontSize',20)
410 ylabel('Pressure [dB]','FontSize',16);xlabel('Frequency [Hz]',' FontSize',16);
411 grid on
412 axis ([0 200 -50 10])
413 legend('Un-Smoothed'); set(Pmic2plot,'linewidth',3)
414
415 Pmic3fft=1/fs*fft(Pmic3,n);
416 figure (300)
417 Pmic3plot=plot(f,(20*log10(Pmic3fft(1:n/2+1))));
418 %title('Frequency Response in Groh Room from a sub-woofer in the
corner with a 10msec pulse','FontSize',20)
419 ylabel('Pressure [dB]','FontSize',16);xlabel('Frequency [Hz]','
FontSize',16);
420 grid on
421 axis ([0 200 -50 10])
422 legend('Un-Smoothed'); set(Pmic3plot,'linewidth',3)
423
424 Pmic4fft=1/fs*fft(Pmic4,n);
425 figure (400)
426 Pmic4plot=plot(f,(20*log10(Pmic4fft(1:n/2+1))));
427 %title('Frequency Response in Groh Room from a sub-woofer in the
corner with a 10msec pulse','FontSize',20)
428 ylabel('Pressure [dB]','FontSize',16);xlabel('Frequency [Hz]','
FontSize',16);
429 grid on
430 axis ([0 200 -50 10])
431 legend('Un-Smoothed'); set(Pmic4plot,'linewidth',3)

%%%Johns schroeder plot
434 Pintegral=0.002; %change this to straighten Schroeder plot
Pschroeder(n+1)=Pintegral;
for i in range(1,n+1):
    Pintegral = Psource(n + 1 - i). ^ 2 + Pintegral;
    Pschroeder(n + 1 - i) = Pintegral;

Pschroeder = 10 * log10(Pschroeder);
figure(11)
441 plot(t,Pschroeder);
442 grid on
443 title('Schroeder decay plot');
444 ylabel('SPL [dB]');xlabel('time (s)');
445 axis ([0 1 -20 70])
446 %%%end of johns shroeder plot


%beginning of my own reverberation method
449 figure (12)
450 Reverbtime(:,1)=t;
451 Reverbtime(:,2)=10*log10(Psource.^2);
452 plot(Reverbtime(:,1),Reverbtime(:,2));
453 axis([0 1 -100 40]);
454 [X Y]=findpeaks(Reverbtime(:,2));
455 Y=Y*K;
456
457 figure (13)
458 plot(Y,X);
459 drawnow
460 %end of my reverberation
461
462 fexpected=1/(pulsewidth*K)
463 load chirp;
464 sound(y,Fs)

"""
figure(4)
surf(P);
title('pressure ');ylabel('time ');xlabel('position ');zlabel('amplitude ') toc
"""

# Thisversion of  Demerit  uses constant band smoothing

tic
load 'PmicarrayGrohReference10cm' flag = 0;
474
bandwidth = 60;
# for now this is just the number of points.IfFres=1 then it is the bandwidth
cuttoff=100;
#this is the cuttoff frequency where the demeritcalculation stops
halfbandwidth=round(bandwidth / 2);
realbandwidth=2 * halfbandwidth+1

for i in range(1,6):
    for j in range(1,6):
        tempequalized = Pmicarray(6 - i, 6 - j,:);
        tempequalized = squeeze(tempequalized);
        tempunequalized = PmicarrayGrohReference10cm(6 - i, 6 - j,:);
        tempunequalized = squeeze(tempunequalized);
        n1 = length(tempequalized) - 1;
        n2 = length(tempunequalized) - 1;

        if n1 ~= n2:
            return
        else:
            n = n1;

        Y=1/fs*fft(tempequalized ,n); # ./transpose(dQfft);
        X=1/fs*fft(tempunequalized ,n); #./transpose(dQfft);

        flag=flag+1;
        f=fs/n*(0:n/2);

#### HASTA LINEA 492

























