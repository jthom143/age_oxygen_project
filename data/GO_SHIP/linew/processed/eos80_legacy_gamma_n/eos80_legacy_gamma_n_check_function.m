%load the eos80_legacy_gamma_n example data
long = 187.317;
lat = -41.6667;

SP =[35.066
35.086
35.089
35.078
35.025
34.851
34.696
34.572
34.531
34.509
34.496
34.452
34.458
34.456
34.488
34.536
34.579
34.612
34.642
34.657
34.685
34.707
34.72
34.729];

t = [12.25
12.21
12.09
11.99
11.69
10.54
9.35
8.36
7.86
7.43
6.87
6.04
5.5
4.9
4.04
3.29
2.78
2.45
2.211
2.011
1.894
1.788
1.554
1.38];

p = [1.0
48.0
97.0
145.0
194.0
291.0
388.0
485.0
581.0
678.0
775.0
872.0
969.0
1066.0
1260.0
1454.0
1647.0
1841.0
2020.0
2216.0
2413.0
2611.0
2878.0
3000.0];

gamma_n_known = [26.6603086452708
    26.6849327992100
    26.7126087875242
    26.7241579596196
    26.7417153499322
    26.8246826743542
    26.9178587846198
    26.9894271737302
    27.0388072986750
    27.0887503278040
    27.1666083192706
    27.2608380621772
    27.3441232325090
    27.4222292032202
    27.5579257860064
    27.6989394498770
    27.7994566049948
    27.8676751857267
    27.9223701628865
    27.9613280531836
    28.0001071113894
    28.0342745352405
    28.0837918317522
    28.1245602716332];

% label the data
[gamma_n, dgl, dgh] = eos80_legacy_gamma_n(SP,t,p,long,lat);

if any(abs(gamma_n_known - gamma_n) > 1e-6)
   fprintf(2,'Your installation of eos80_legacy_gamma_n has errors !\n');
else
    fprintf(1,'The eos80 legacy gamma_n check fuctions confirms that the \n');
    fprintf(1,'eos80_legacy_gamma_n is installed correctly.\n');
end

SP_ns_known =[34.905417438614279
  34.629759136980965
  34.723585265547896];

t_ns_known = [10.899824105694968
   2.308533246341538
   1.484693189324784];

p_ns_known = [ 260.663841692037
   1946.962850653058
   2926.600266315955];
   
% fit three surfaces
gamma_n_surfaces = [26.8, 27.9, 28.1];
[SP_ns,t_ns,p_ns] = eos80_legacy_neutral_surfaces(SP,t,p,gamma_n_known,gamma_n_surfaces);

if any(abs(SP_ns - SP_ns_known) > 1e-6) | any(abs(t_ns - t_ns_known) > 1e-6) | any(abs(p_ns - p_ns_known) > 1e-2)
   fprintf(2,'Your installation of eos80_legacy_neutral_surfaces has errors !\n');
else
    fprintf(1,'The eos80 legacy gamma_n check fuctions confirms that the \n');
    fprintf(1,'eos80_legacy_neutral_surfaces is installed correctly.\n');
    fprintf(1,'Well done! the EOS80 version of the Neutral Density (gamma_n) code is now ready for use.\n');
end

clear SP t p long lat gamma_n_known gamma_n  dgl dgh SP_ns_known t_ns_known p_ns_known gamma_n_surfaces SP_ns t_ns p_ns
