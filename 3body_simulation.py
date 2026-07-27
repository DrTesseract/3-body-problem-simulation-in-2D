import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Working directory of the Python file
file_path = os.path.dirname(__file__)
os.chdir(file_path)


#------------------------
#masses of 3 bodys
m1=1
m2=1
m3=1

#mass of the planet 
mp=0.001

# consider 3-body problem with planet in the 2d plane
# the representative vector of the state of the system is defined as
# Y=[x1, y1, x2, y2, x3, y3,  xp, yp, px1, py1, px2, py2, px3, py3, pxp, pyp]


# for the 3-body problem, the derivative vector dY/dt=F(Y) = (f1,f2,..) for each state
def F(state): 

    if len(state)== 16: #consideration of planet, state= [x1, y1, x2, y2, x3, y3,  xp, yp, px1, py1, px2, py2, px3, py3, pxp, pyp]

        
        # the current state Y
        x1, y1, x2, y2, x3, y3, xp, yp, px1, py1, px2, py2, px3, py3, pxp, pyp = state
        # then for the first 8 elements holds: dxi/dt = pxi/mi , dyi/dt = pyi/mi,...
        f1 = px1/m1
        f2 = py1/m1
        f3 = px2/m2
        f4 = py2/m2
        f5 = px3/m3
        f6 = py3/m3
        f7 = pxp/mp
        f8 = pyp/mp
            
        
        # for the remaining fi holds: dpi/dt = - grad_i(U), with the potential U of the 3 bodies by gravitation 
        # a simple calculation of the derivatives yields the following fi
        
        # Distances of all 4 bodies to each other
        abs12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        abs13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
        abs23 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
        abs1p = np.sqrt((x1-xp)**2 + (y1-yp)**2)
        abs2p = np.sqrt((x2-xp)**2 + (y2-yp)**2)
        abs3p = np.sqrt((x3-xp)**2 + (y3-yp)**2)
        
        #for the bodys
        f9  = -m1*m2*(x1-x2)/abs12**3 - m1*m3*(x1-x3)/abs13**3 - m1*mp*(x1-xp)/abs1p**3 # dpx1/dt = -dU/dx1
        f10 = -m1*m2*(y1-y2)/abs12**3 - m1*m3*(y1-y3)/abs13**3 - m1*mp*(y1-yp)/abs1p**3 # dpy1/dt = -dU/dy1
        
        f11 = -m1*m2*(x2-x1)/abs12**3 - m2*m3*(x2-x3)/abs23**3 - m2*mp*(x2-xp)/abs2p**3 # dpx2/dt = -dU/dx2
        f12 = -m1*m2*(y2-y1)/abs12**3 - m2*m3*(y2-y3)/abs23**3 - m2*mp*(y2-yp)/abs2p**3 # dpy2/dt = -dU/dy2
        
        f13 = -m1*m3*(x3-x1)/abs13**3 - m2*m3*(x3-x2)/abs23**3 - m3*mp*(x3-xp)/abs3p**3 # dpx3/dt = -dU/dx3
        f14 = -m1*m3*(y3-y1)/abs13**3 - m2*m3*(y3-y2)/abs23**3 - m3*mp*(y3-yp)/abs3p**3 # dpy3/dt = -dU/dy3
        
        # for the planet (momentum derivatives influenced by the 3 bodies)
        f15 = -m1*mp*(xp-x1)/abs1p**3 - m2*mp*(xp-x2)/abs2p**3 - m3*mp*(xp-x3)/abs3p**3 # dpxp/dt = -dU/dxp
        f16 = -m1*mp*(yp-y1)/abs1p**3 - m2*mp*(yp-y2)/abs2p**3 - m3*mp*(yp-y3)/abs3p**3 # dpyp/dt = -dU/dyp
        
    
        return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12, f13, f14, f15, f16])

      
    if len(state) == 12: # if we dont have a planet 
        x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = state
        
        f1 = px1/m1
        f2 = py1/m1
        f3 = px2/m2
        f4 = py2/m2
        f5 = px3/m3
        f6 = py3/m3
        
        abs12 = np.sqrt((x1-x2)**2+(y1-y2)**2)
        abs13 = np.sqrt((x1-x3)**2+(y1-y3)**2)
        abs23 = np.sqrt((x2-x3)**2+(y2-y3)**2)
        
        f7 = -m1*m2*(x1-x2)/abs12**3 - m1*m3*(x1-x3)/abs13**3
        f8 = -m1*m2*(y1-y2)/abs12**3 - m1*m3*(y1-y3)/abs13**3
        f9 = -m1*m2*(x2-x1)/abs12**3 - m2*m3*(x2-x3)/abs23**3
        f10 = -m1*m2*(y2-y1)/abs12**3 - m2*m3*(y2-y3)/abs23**3
        f11 = -m1*m3*(x3-x1)/abs13**3 - m2*m3*(x3-x2)/abs23**3
        f12 = -m1*m3*(y3-y1)/abs13**3 - m2*m3*(y3-y2)/abs23**3
        
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
         
    
    


# with Euler method for a given state Y_i, calculate the next Y_(i+1)
def euler(state, h=0.01):
    # step size h
    state_new = state + h*F(state)
    return state_new    

    
# improved Euler method
def euler_improved(state, h=0.01):
    state_new = state + h*F(state+h*F(state)/2)
    return state_new

    
# Runge-Kutta method order 5 (automatically finds best parameters s, aij, bj, cj) 
def RK(state, t, h=0.01):
    sol = solve_ivp(lambda t, y: F(y), (t,t+h), state, method="RK45", t_eval=[t+h] )
    return sol.y[:,-1]
    

def energy(state):
    # Fall 1: Reines 3-Körper-Problem (12 Elemente, kein Planet)
    if len(state) == 12:
        x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = state
        
        E_pot = -m1*m2/np.sqrt((x1-x2)**2+(y1-y2)**2) - m1*m3/np.sqrt((x1-x3)**2+(y1-y3)**2) - m2*m3/np.sqrt((x2-x3)**2+(y2-y3)**2)
        E_kin = 2*(px1**2+py1**2)/m1 +  2*(px2**2+py2**2)/m2 +  2*(px3**2+py3**2)/m3
        
        return E_kin + E_pot
        
    # Fall 2: 4-Körper-Problem mit Planet (16 Elemente)
    else:
        x1, y1, x2, y2, x3, y3, xp, yp, px1, py1, px2, py2, px3, py3, pxp, pyp = state
        
        E_pot = -m1*m2/np.sqrt((x1-x2)**2+(y1-y2)**2) - m1*m3/np.sqrt((x1-x3)**2+(y1-y3)**2) - m2*m3/np.sqrt((x2-x3)**2+(y2-y3)**2)
        E_pot += -m1*mp/np.sqrt((x1-xp)**2+(y1-yp)**2) - m2*mp/np.sqrt((x2-xp)**2+(y2-yp)**2) - m3*mp/np.sqrt((x3-xp)**2+(y3-yp)**2)

        # Schutz vor Division durch Null, falls mp auf 0 gesetzt wurde
        E_kin_planet = 2*(pxp**2+pyp**2)/mp if mp != 0 else 0.0
        E_kin = 2*(px1**2+py1**2)/m1 +  2*(px2**2+py2**2)/m2 +  2*(px3**2+py3**2)/m3 + E_kin_planet
        
        return E_kin + E_pot

#----------------------------

# the function that creates the animation of the 3body movement and saves it to a mp4 file
#it is designed to eather compare the 3 algorithms with each other
#or to fix one algorithm and also observe the movement of a planet in the 3 body system


def make_animation(filename, initialStateVector, masses = [1,1,1, 0.001], animation_name="3 Body Problem Simulation", animation_size = (8,8) ,plot=True, trail=True, methods=[0,0,1], frame_count=2000, frame_interval=1, show_energy=True, planet = False, planet_method=2,  h_d = 0.01):
    
    """
    -takes filename "example.mp4"
    -takes initial state vector of the 3 bodys Y=[x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3] positions and momentum
    if we consider the planet then the vector has length 16 and the shape
    [x1, y1, x2, y2, x3, y3,  xp, yp, px1, py1, px2, py2, px3, py3, pxp, pyp]
    
    -set the masses = [m1,m2,m3, mp]
    -animation size is the tuple which determines figure size
    -Number of frames frame_count and frame_interval in ms
    -if plot=True only a plot is output, the file is not saved and vice versa
    -methods =[euler, imp_euler, RK] select which ones should all be used
    -trail=True if one wants to have the trails
    -show_energy=True, if one wants to have the energies of the bodies in the plot (physically the should be constant, but they change due to approximation errors)
    -h_d is the parameter (step size) of the euler/RK method
    -planet= True adds a planet
    
    """
    #if needed the change the global masses
    global m1, m2, m3, mp  
    m1, m2, m3, mp = masses
    
    # prepare figure and ax 
    fig, ax = plt.subplots(figsize=animation_size)
    ax.set_xlim(-4, 4) 
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(animation_name)
    
    # set background to dark blue
    ax.set_facecolor("#0a1128")
    
    # stars in the background (¬‿¬)
    star_count = 200
    stars_x = np.random.uniform(-4, 4, star_count)
    stars_y = np.random.uniform(-4, 4, star_count)
    brightnesses = np.random.uniform(0.1, 0.9, star_count)
    sizes = np.random.uniform(0.1, 2.5, star_count)
    
    ax.scatter(stars_x, stars_y, s=sizes, c='white', alpha=brightnesses, zorder=0)
        
    # current state vector Y (create copies so they are independent)
    Y_eu = initialStateVector.copy()
    Y_eui = initialStateVector.copy()
    Y_rk = initialStateVector.copy()
    
    # start time 0
    t_d = 0
        
    
    
    # only create the suns if the method was actually selected
    if methods[0] == 1: #simple euler
        # suns as line2D object, a point in the plot will always be updated
        sun1, = ax.plot([],[], "o",color="white", markeredgecolor="#00ffff", markeredgewidth=3, markersize=9+np.log(m1), label="Euler Method")
        sun2, = ax.plot([],[], "o",color="white", markeredgecolor="#00aaff",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3, = ax.plot([],[], "o",color="white", markeredgecolor="#5555ff",  markeredgewidth=3, markersize=9+np.log(m3))
        
        # planet for simple euler
        if planet and planet_method == 0:
            planet_eu, = ax.plot([],[], "o",color="white", markeredgecolor="lime", markeredgewidth=2, markersize=8.5+np.log(max(mp, 1e-10)), label="Planet (Euler)")
            trail_p_eu = []
        
        # I will always save all calculated Y in a list
        Y_list_eu = []
        
        # trails of the suns are lists of plots (line2D objects) with different alpha
        trail1_eu, trail2_eu, trail3_eu = [], [], []

    if methods[1] == 1: #better euler
        sun1ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ffff99", markeredgewidth=3, markersize=9+np.log(m1), label= "imp. Euler Method")
        sun2ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ffd700",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ff9900",  markeredgewidth=3, markersize=9+np.log(m3))
        
        # planet for imp. euler
        if planet and planet_method == 1:
            planet_eui, = ax.plot([],[], "o",color="white", markeredgecolor="lime", markeredgewidth=2, markersize=8.5+np.log(max(mp, 1e-10)), label="Planet (imp. Euler)")
            trail_p_eui = []
            
        Y_list_eui = []
        trail1_eui, trail2_eui, trail3_eui = [], [], []

    if methods[2] == 1:
        sun1rk, = ax.plot([],[], "o",color="white", markeredgecolor="#ff7777", markeredgewidth=3, markersize=9+np.log(m1), label= "Runge Kutta Method")
        sun2rk, = ax.plot([],[], "o",color="white", markeredgecolor="#ff3333",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3rk, = ax.plot([],[], "o",color="white", markeredgecolor="#cc0055",  markeredgewidth=3, markersize=9+np.log(m3))
        
        # planet for RK
        if planet and planet_method == 2:
            planet_rk, = ax.plot([],[], "o",color="white", markeredgecolor="lime", markeredgewidth=2, markersize=8.5+np.log(max(mp, 1e-10)), label="Planet (RK)")
            trail_p_rk = []
            
        Y_list_rk = []
        trail1_rk, trail2_rk, trail3_rk = [], [], []
        
    
    # for the trails this determines the length of the trail
    segment_count = 100 
    
    if trail == True:
        for i in range(segment_count):
            # create for each pass of the loop new larger alpha (lower transparency)
            alpha_value = 0.12 * i / segment_count
            
            # again separated respectively for the 3 methods
            if methods[0] == 1:
                # generate line plots with different alpha
                s1, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="blue")
                s2, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="blue")
                s3, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="blue")
                
                # insert these line2d objects into the lists--> many line segments of different transparency
                trail1_eu.append(s1); trail2_eu.append(s2); trail3_eu.append(s3)
                
                if planet and planet_method == 0:
                    sp_eu, = ax.plot([], [], "-", alpha=alpha_value, lw=1.5, color="white")
                    trail_p_eu.append(sp_eu)
        
            if methods[1] == 1:
                s1ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                s2ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                s3ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                trail1_eui.append(s1ie); trail2_eui.append(s2ie); trail3_eui.append(s3ie)
                
                if planet and planet_method == 1:
                    sp_eui, = ax.plot([], [], "-", alpha=alpha_value, lw=1.5, color="white")
                    trail_p_eui.append(sp_eui)
            
            if methods[2] == 1:
                s1rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                s2rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                s3rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                trail1_rk.append(s1rk); trail2_rk.append(s2rk); trail3_rk.append(s3rk)
                
                if planet and planet_method == 2:
                    sp_rk, = ax.plot([], [], "-", alpha=alpha_value, lw=1.5, color="white")
                    trail_p_rk.append(sp_rk)
    # display energies
    if show_energy==True:
        E_real = energy(initialStateVector)
        ax.text(0.05, 0.95, f"Energy (Physics): {E_real:.4f}", transform=ax.transAxes, color="white", fontsize=12)
        if methods[0]==1:
            energy_text_eu = ax.text(0.05, 0.80, "", transform=ax.transAxes, color="white", fontsize=12)
        if methods[1]==1:
            energy_text_eui = ax.text(0.05, 0.85, "", transform=ax.transAxes, color="white", fontsize=12)
        if methods[2]==1:
            energy_text_rk = ax.text(0.05, 0.90, "", transform=ax.transAxes, color="white", fontsize=12)
        
   # now the update function, which updates the sun positions and trail segment positions
    def update(i):
        # also use variables from above
        nonlocal Y_eu, Y_eui, Y_rk, t_d
        # new time step
        t_d = t_d + h_d
        
        # Here we collect all objects that have updated (plots with new x,y)
        updated_objects = [] 
        
        
        # --- EULER ---
        if methods[0] == 1:
        
            # new updated state vector Y
            Y_eu = euler(Y_eu, h_d)
            # update xy and y values of the 3 bodies in the plots
            sun1.set_data([Y_eu[0]], [Y_eu[1]])
            sun2.set_data([Y_eu[2]], [Y_eu[3]])
            sun3.set_data([Y_eu[4]], [Y_eu[5]])
            updated_objects.extend([sun1, sun2, sun3])
            
            # update planet position if active for this method
            if planet and planet_method == 0:
                planet_eu.set_data([Y_eu[6]], [Y_eu[7]])
                updated_objects.append(planet_eu)
            
            # prepare trails
            if trail == True:
                # always collect previous Y state vectors in a list
                Y_list_eu.append(Y_eu)
                
                # if list longer than number of segments in the trail, delete the zeroth element-> save memory 
                if len(Y_list_eu) > segment_count + 1:
                    Y_list_eu.pop(0)
                    
                # last n(number of segments) positions of the state list as array    
                lnp = np.array(Y_list_eu)
                
                if len(lnp) > 1:
                    # update positions of the trail segments, segment of highest transparency is the last 
                    for j in range(min(segment_count, len(lnp) - 1)):
                        trail1_eu[j].set_data(lnp[j:j+2,0], lnp[j:j+2,1]) # extract x1 and y1 values from lnp, sequentially take as positions of the line segments
                        trail2_eu[j].set_data(lnp[j:j+2,2], lnp[j:j+2,3])  # extract x2 and y2 values from lnp,..
                        trail3_eu[j].set_data(lnp[j:j+2,4], lnp[j:j+2,5])  # extract x3 and y13 values from lnp,..
                        
                        if planet and planet_method == 0:
                            trail_p_eu[j].set_data(lnp[j:j+2,6], lnp[j:j+2,7]) # extract xp and yp values from lnp
                            
                    updated_objects.extend(trail1_eu + trail2_eu + trail3_eu)
                    if planet and planet_method == 0:
                        updated_objects.extend(trail_p_eu)
                
            if show_energy==True:
                E_euler= energy(Y_eu)
                energy_text_eu.set_text(f"Energy (Euler): {E_euler:.4f}")                
                updated_objects.append(energy_text_eu)
                
        # --- IMPROVED EULER ---
        if methods[1] == 1:
            Y_eui = euler_improved(Y_eui, h_d)
            sun1ie.set_data([Y_eui[0]], [Y_eui[1]])
            sun2ie.set_data([Y_eui[2]], [Y_eui[3]])
            sun3ie.set_data([Y_eui[4]], [Y_eui[5]])
            updated_objects.extend([sun1ie, sun2ie, sun3ie])
            
            # update planet position if active for this method
            if planet and planet_method == 1:
                planet_eui.set_data([Y_eui[6]], [Y_eui[7]])
                updated_objects.append(planet_eui)
            
            if trail == True:
                Y_list_eui.append(Y_eui)
                if len(Y_list_eui) > segment_count + 1:
                    Y_list_eui.pop(0)
                    
                lnp_eui = np.array(Y_list_eui)
                if len(lnp_eui) > 1:
                    for j in range(min(segment_count, len(lnp_eui) - 1)):
                        trail1_eui[j].set_data(lnp_eui[j:j+2,0], lnp_eui[j:j+2,1])
                        trail2_eui[j].set_data(lnp_eui[j:j+2,2], lnp_eui[j:j+2,3])
                        trail3_eui[j].set_data(lnp_eui[j:j+2,4], lnp_eui[j:j+2,5])
                        
                        if planet and planet_method == 1:
                            trail_p_eui[j].set_data(lnp_eui[j:j+2,6], lnp_eui[j:j+2,7])
                            
                    updated_objects.extend(trail1_eui + trail2_eui + trail3_eui)
                    if planet and planet_method == 1:
                        updated_objects.extend(trail_p_eui)
                
            if show_energy==True:
                E_euler_improved= energy(Y_eui)
                energy_text_eui.set_text(f"Energy (imp. Euler): {E_euler_improved:.4f}")                
                updated_objects.append(energy_text_eui)    
                
        # --- RUNGE KUTTA ---
        if methods[2] == 1:
            Y_rk = RK(Y_rk, t_d, h_d)
            sun1rk.set_data([Y_rk[0]], [Y_rk[1]])
            sun2rk.set_data([Y_rk[2]], [Y_rk[3]])
            sun3rk.set_data([Y_rk[4]], [Y_rk[5]])
            updated_objects.extend([sun1rk, sun2rk, sun3rk])
            
            # update planet position if active for this method
            if planet and planet_method == 2:
                planet_rk.set_data([Y_rk[6]], [Y_rk[7]])
                updated_objects.append(planet_rk)
            
            if trail == True:
                Y_list_rk.append(Y_rk)
                if len(Y_list_rk) > segment_count + 1:
                    Y_list_rk.pop(0)
                    
                lnp_rk = np.array(Y_list_rk)
                if len(lnp_rk) > 1:
                    for j in range(min(segment_count, len(lnp_rk) - 1)):
                        trail1_rk[j].set_data(lnp_rk[j:j+2,0], lnp_rk[j:j+2,1])
                        trail2_rk[j].set_data(lnp_rk[j:j+2,2], lnp_rk[j:j+2,3])
                        trail3_rk[j].set_data(lnp_rk[j:j+2,4], lnp_rk[j:j+2,5])
                        
                        if planet and planet_method == 2:
                            trail_p_rk[j].set_data(lnp_rk[j:j+2,6], lnp_rk[j:j+2,7])
                            
                    updated_objects.extend(trail1_rk + trail2_rk + trail3_rk)
                    if planet and planet_method == 2:
                        updated_objects.extend(trail_p_rk)
            
            if show_energy==True:
                E_rk= energy(Y_rk)
                energy_text_rk.set_text(f"Energy (Runge Kutta): {E_rk:.4f}")                
                updated_objects.append(energy_text_rk)
        
        # a single return for all active objects
        return updated_objects



    # create animation
    print("Starting animation...")
    ani = animation.FuncAnimation(
        fig,                  
        update,             
        frames=frame_count,            
        interval=frame_interval,          
        blit=True             
    )   
        
    ax.grid()
    # Only create a legend if plots are actually present
    if any(methods): 
        ax.legend(loc="upper right", framealpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if plot == True:
        # plot animation, terminate function
        return ani
    else:
        # do not plot animation, export video
        print("Saving video... ")
        FFwriter = animation.FFMpegWriter(fps=50)
        ani.save(filename, writer=FFwriter)
        print("Finished saving!")
        plt.close(fig)




#examples



"""

#example 
#only runge kutta, dont save mp4  just show, with planet, chaotic system
Y_ex1 = np.array([-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0,0,-0.4 ,0.2]+
                [0.4662036850, 0.4323657300, 0.4662036850, 0.4323657300, -0.93240737, -0.86473146,-0.000001,-0.00002 ])



eight_orbit = make_animation("Trisolaris.mp4", Y_ex1, masses= [1.01,1.1,1,0.001] ,plot=False, animation_name="Simulation of Trisolaris", methods=[0,0,1], frame_count=2000, planet= True)

"""








# figure-eight orbit, comparison of all three algorithms, without planet

Y_0 = Y_ex1 = np.array([-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0,0]+
                [0.4662036850, 0.4323657300, 0.4662036850, 0.4323657300, -0.93240737, -0.86473146])

eight_orbit = make_animation("algorithm_comparison.mp4", Y_0, masses=[1, 1, 1, 0], plot=False, methods=[1, 1, 1], frame_count=2000, planet=False)








"""

#only runge kutta, only three bodys
# Triangle, rotation with distance 1 around center of mass at 0,0 (only Runge Kutta)
Y_triangle_stat = np.array([0,-1,np.cos(np.pi/6),np.sin(np.pi/6),-np.cos(np.pi/6),np.sin(np.pi/6),0.5,0,-np.sin(np.pi/6)*0.5,np.cos(np.pi/6)*0.5,-np.sin(np.pi/6)*0.5,-np.cos(np.pi/6)*0.5])

triangle_stat = make_animation("3_body_Triangle.mp4", Y_triangle_stat, plot= True, methods = [0,0,1], frame_count=9000, frame_interval=0.5 )
"""













