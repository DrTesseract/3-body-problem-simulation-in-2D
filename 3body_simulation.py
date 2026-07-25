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

# consider 3-body problem in the 2d plane
# the representative vector of the state of the system is defined as
# Y=[x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3]


# for the 3-body problem, the derivative vector dY/dt=F(Y) = (f1,f2,..) for each state
def F(state): 
    # the current state Y
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = state
    # then for the first 6 elements holds: dxi/dt = pxi/mi , dyi/dt = pyi/mi,...
    f1 = px1/m1
    f2 = py1/m1
    f3 = px2/m2
    f4 = py2/m2
    f5 = px3/m3
    f6 = py3/m3
    
    # for the remaining fi holds: dpi/dt = - grad_i(U), with the potential U of the 3 bodies by gravitation
    # a simple calculation of the derivatives yields the following fi
    
    abs12 = np.sqrt((x1-x2)**2+(y1-y2)**2)
    abs13 = np.sqrt((x1-x3)**2+(y1-y3)**2)# Distances of the 3 bodies to each other
    abs23 = np.sqrt((x2-x3)**2+(y2-y3)**2)
    
    
    
    f7 = -m1*m2*(x1-x2)/abs12**3 -m1*m3*(x1-x3)/abs13**3 # dpx1/dt= -dU/dx1
    f8 = -m1*m2*(y1-y2)/abs12**3 -m1*m3*(y1-y3)/abs13**3 # dpy1/dt= -dU/dy1
    f9 = -m1*m2*(x2-x1)/abs12**3 -m2*m3*(x2-x3)/abs23**3 # dpx2/dt= -dU/dx2
    f10 = -m1*m2*(y2-y1)/abs12**3 -m2*m3*(y2-y3)/abs23**3 # dpy2/dt= -dU/dy2
    f11 = -m1*m3*(x3-x1)/abs13**3 -m2*m3*(x3-x2)/abs23**3 # dpx3/dt= -dU/dx3
    f12 = -m1*m3*(y3-y1)/abs13**3 -m2*m3*(y3-y2)/abs23**3 # dpy3/dt= -dU/dy3

    return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])


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
    


# function which determines the energy of each state
def energy(state):
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = state
    
    E_pot = -m1*m2/np.sqrt((x1-x2)**2+(y1-y2)**2) -m1*m3/np.sqrt((x1-x3)**2+(y1-y3)**2)-m2*m3/np.sqrt((x2-x3)**2+(y2-y3)**2)

    E_kin = 2*(px1**2+py1**2)/m1 +  2*(px2**2+py2**2)/m2 +  2*(px3**2+py3**2)/m3
    return E_kin +E_pot



#----------------------------

# the function that creates the animation of the 3body movement and saves it to a mp4 file

def make_animation(filename, initialStateVector, masses = [1,1,1], animation_name="3 Body Problem Simulation", plot=False, trail=True, methods=[1,1,1], frame_count=2000, frame_interval=1, show_energy=True, h_d = 0.01):
    
    """
    -takes filename "example"
    -takes initial state vector of the 3 bodys Y=[x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3] positions and momentum
    -set the masses = [m1,m2,m3]
    -Number of frames frame_count and frame_interval in ms
    -if plot=True only a plot is output, the file is not saved and vice versa
    -methods =[euler, imp_euler, RK] select which ones should all be used
    -trail=True if one wants to have the trails
    -show_energy=True, if one wants to have the energies of the bodies in the plot (physically the should be constant, but they change due to approximation errors)
    -h_d is the parameter (step size) of the euler/RK method
    
    """
    #if needed the change the global masses
    global m1, m2, m3  
    m1, m2, m3 = masses
    
    # prepare figure and ax 
    fig, ax = plt.subplots(figsize=(8,8))
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
    if methods[0] == 1:
        # suns as line2D object, a point in the plot will always be updated
        sun1, = ax.plot([],[], "o",color="white", markeredgecolor="#00ffff", markeredgewidth=3, markersize=9+np.log(m1), label="Euler Method")
        sun2, = ax.plot([],[], "o",color="white", markeredgecolor="#00aaff",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3, = ax.plot([],[], "o",color="white", markeredgecolor="#5555ff",  markeredgewidth=3, markersize=9+np.log(m3))
        
        # I will always save all calculated Y in a list
        Y_list_eu = []
        
        # trails of the suns are lists of plots (line2D objects) with different alpha
        trail1_eu, trail2_eu, trail3_eu = [], [], []

    if methods[1] == 1:
        sun1ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ffff99", markeredgewidth=3, markersize=9+np.log(m1), label= "imp. Euler Method")
        sun2ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ffd700",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3ie, = ax.plot([],[], "o",color="white", markeredgecolor="#ff9900",  markeredgewidth=3, markersize=9+np.log(m3))
        Y_list_eui = []
        trail1_eui, trail2_eui, trail3_eui = [], [], []

    if methods[2] == 1:
        sun1rk, = ax.plot([],[], "o",color="white", markeredgecolor="#ff7777", markeredgewidth=3, markersize=9+np.log(m1), label= "Runge Kutta Method")
        sun2rk, = ax.plot([],[], "o",color="white", markeredgecolor="#ff3333",  markeredgewidth=3, markersize=9+np.log(m2))
        sun3rk, = ax.plot([],[], "o",color="white", markeredgecolor="#cc0055",  markeredgewidth=3, markersize=9+np.log(m3))
        Y_list_rk = []
        trail1_rk, trail2_rk, trail3_rk = [], [], []
        
    
    # for the trails this determines the length    
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
        
            if methods[1] == 1:
                s1ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                s2ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                s3ie, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="yellow")
                trail1_eui.append(s1ie); trail2_eui.append(s2ie); trail3_eui.append(s3ie)
            
            if methods[2] == 1:
                s1rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                s2rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                s3rk, = ax.plot([], [], "-", alpha=alpha_value, lw=3, color="red")
                trail1_rk.append(s1rk); trail2_rk.append(s2rk); trail3_rk.append(s3rk)

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
                updated_objects.extend(trail1_eu + trail2_eu + trail3_eu)
                
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
                updated_objects.extend(trail1_eui + trail2_eui + trail3_eui)
                
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
                updated_objects.extend(trail1_rk + trail2_rk + trail3_rk)
            
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




















# figure-eight orbit (all 3 methods in comparison), dont save mp4  just show
Y_0 = np.array([-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0, 0]+
                [0.4662036850, 0.4323657300, 0.4662036850, 0.4323657300, -0.93240737, -0.86473146 ])

eight_orbit = make_animation("shorttest.mp4", Y_0, plot=True, trail=True, methods=[0,0,1], frame_count=40, h_d=0.01)





# Further stable solutions and examples


"""
# Triangle, rotation with distance 1 around center of mass at 0,0 (only Runge Kutta)
Y_triangle_stat = np.array([0,-1,np.cos(np.pi/6),np.sin(np.pi/6),-np.cos(np.pi/6),np.sin(np.pi/6),0.5,0,-np.sin(np.pi/6)*0.5,np.cos(np.pi/6)*0.5,-np.sin(np.pi/6)*0.5,-np.cos(np.pi/6)*0.5])

triangle_stat = make_animation("3_body_Triangle.mp4", Y_triangle_stat, plot= True, methods = [0,0,1], frame_count=9000, frame_interval=0.5 )
"""


"""
# Sun-Earth-Moon
m1 = 100   # Sun
m2 = 1# Earth
m3 = 0.01

v_earth = 0.9*np.sqrt(100.0 / 2.5)                

# The moon moves much closer to the earth (distance 0.15 instead of 0.4)
# Because of this it must orbit much faster, so as not to fall onto the earth!
v_moon_rel = np.sqrt(1.0 / 0.15)              

# Absolute velocity
v_moon_abs = v_earth + v_moon_rel            

# Conversion into momenta
p_y2 = 1.0 * v_earth                          
p_y3 = 0.01 * v_moon_abs                      
p_y1 = -(p_y2 + p_y3)                        

# The corrected state vector
Y_sem = np.array([
    0.0, 0.0,        # Sun
    2.5, 0.0,        # Earth
    2.65, 0.0,       # NEW: Moon is at x=2.65 (only 0.15 away from the Earth)
    0.0, p_y1,        
    0.0, p_y2,        
    0.0, p_y3        
])



m = make_animation("hh", Y_sem, plot= True, methods = [0,0,1])
"""










