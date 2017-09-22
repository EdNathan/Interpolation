import numpy as np
import astropy.cosmology as apcos
from astropy import units as u
from astropy import constants as const

cosmo = apcos.WMAP7 



def interpolate_cen_cen(xi, yi, zi, vxi, vyi, vzi, xf, yf, zf, vxf, vyf, vzf, z_red_i, z_red_f, z_red_interpolate):
    '''
    xi, yi, zi are resecptively all arrays of the initial x,y,z coordinates of each galaxy.
    xf, yf, zf are resecptively all arrays of the final x,y,z coordinates of each galaxy.
    vxi, vyi, vzi are resecptively all arrays of the initial velocity in the x,y,z directions, for each galaxy.
    vxf, vyf, vzf are resecptively all arrays of the final velocity in the x,y,z directions, for each galaxy.
    z_red_i, z_red_f are respectively the initial and final redshifts of the galaxies.
    z_red_interpolate is an array of the redshifts for interpolation, and should be between z_red_i and z_red_f.
    
    All galaxys should have the same initial and final times, and will be interpolated at the same locations.
    
    Array is structred so that first index is the snapshot, second index is the galaxy, 3rd index is position/velocity vector.
    If only one galaxy (or one time, or both) is provided, that index is omitted from the results.
    
    Note:  Units need to be Mpc/h for distances, and km/s for vecloties'''
    
    
    # Unit conversion:
    #   x/y/zgal   : comoving position in Mpc/h
    #   vx/vy/vzgal: peculiar velocity in km/sec
    #   
    #   Need to convert from peculiar velocities in km/sec
    #   to comoving velocities in (Mpc/h)/Gyr. Will then fit
    #   cubic function to x(t) in comoving coordinates.
    
    
    
    time_interp = cosmo.age(z_red_interpolate)
    ti, tf = cosmo.age([z_red_i, z_red_f])
        
    tscaled = (time_interp - ti) / (tf - ti)
  
    
    # Speed is initially provided as km/s.
    unit_scale = ((tf-ti).to(u.s) * u.km / u.s).to(u.Mpc).value  
    
    #Scale the units - velocities are now comoving.    
    xinit = np.array([xi, yi, zi]).T / cosmo.h
    xfin  = np.array([xf, yf, zf]).T / cosmo.h
    vinit = np.array([vxi, vyi, vzi]).T * unit_scale / cosmo.scale_factor(z_red_i)
    vfin  = np.array([vxf, vyf, vzf]).T * unit_scale / cosmo.scale_factor(z_red_f)

    
    # Rescale t so that t1 = 0 and t2 = 1, then
    # A = x(t1)
    # B = v(t1)
    # C = -2*v(t1) - 3*x(t1) - v(t2) + 3*x(t2)
    # D = v(t1) + v(t2) + 2*(x(t1)-x(t2))
    
    # x(t) = A + B*t + C*t**2 + D*t**3
    # v(t) = B + 2*C*t + 3*D*t**2  
    
     
    A = xinit
    B = vinit
    C = -2*vinit - 3*xinit - vfin + 3*xfin
    D = vinit + vfin + 2*(xinit-xfin)

    #print B.shape
    #print tscaled.shape
    
    
    pos_interp = np.add(A, np.tensordot(tscaled,B,axes=0) + 
                        np.tensordot(np.power(tscaled,2),C,axes=0) + np.tensordot(np.power(tscaled,3),D,axes=0))
    
    vel_interp = np.add(B, 2*np.tensordot(tscaled,C,axes=0) + 
                        3*np.tensordot(np.power(tscaled,2),D,axes=0))
     
    
    #rescale and return times, position, and velocity.     
    return time_interp.value, pos_interp*cosmo.h, (vel_interp.T * cosmo.scale_factor(z_red_interpolate)).T/ unit_scale
        

def time_spaced_interpolation_cen_cen(xi, yi, zi, vxi, vyi, vzi, xf, yf, zf, vxf, vyf, vzf, z_red_i, z_red_f, 
                                      samples=50, return_redshifts=False):
    '''
    xi, yi, zi are resecptively all arrays of the initial x,y,z coordinates of each galaxy.
    xf, yf, zf are resecptively all arrays of the final x,y,z coordinates of each galaxy.
    vxi, vyi, vzi are resecptively all arrays of the initial velocity in the x,y,z directions, for each galaxy.
    vxf, vyf, vzf are resecptively all arrays of the final velocity in the x,y,z directions, for each galaxy.
    z_red_i, z_red_f are respectively the initial and final redshifts of the galaxies.
    
    samples is the number of samples the code uses between the initial and final redshidfts, spread evenly in time.
    return_redshifts causes the redshifts of the samples taken to be returned.
    
    All galaxys should have the same initial and final times, and will be interpolated at the same locations.
    

    
    Note:  Units need to be Mpc/h for distances, and km/s for vecloties
    '''
    ti, tf = cosmo.age([z_red_i, z_red_f])
    ages = np.linspace(ti, tf, samples, endpoint=True)
    redshifts = np.zeros(samples)
    
    #Must be a better way to do this?
    for i, age in enumerate(ages):
        redshifts[i] = apcos.z_at_value(cosmo.age, age)
    
    times, pos, vel = interpolate_cen_cen(xi, yi, zi, vxi, vyi, vzi, xf, yf, zf, 
                                          vxf, vyf, vzf, z_red_i, z_red_f, redshifts)
    
    if return_redshifts:
        return times, pos, vel, redshifts
    else:
        return times, pos, vel
    

def interpolate_sat_sat(sat_xi, sat_yi, sat_zi, sat_vxi, sat_vyi, sat_vzi, 
                        sat_xf, sat_yf, sat_zf, sat_vxf, sat_vyf, sat_vzf, 
                        cen_xi, cen_yi, cen_zi, cen_vxi, cen_vyi, cen_vzi, 
                        cen_xf, cen_yf, cen_zf, cen_vxf, cen_vyf, cen_vzf, 
                        z_red_i, z_red_f, z_red_interpolate):
    '''
    Important note:  There will be an issue in the case that one/both of the positions is at the central galaxy,
    within the avaliable precision.
    '''
            
    # Interpolate central galazy
    times, cen_pos, cen_vel = interpolate_cen_cen(cen_xi, cen_yi, cen_zi, cen_vxi, cen_vyi, cen_vzi,
                                                  cen_xf, cen_yf, cen_zf, cen_vxf, cen_vyf, cen_vzf, 
                                                  z_red_i, z_red_f, z_red_interpolate)
    
    # Convert units, and make comoving
    time_interp = cosmo.age(z_red_interpolate)
    ti, tf = cosmo.age([z_red_i, z_red_f])
        
    tscaled = (time_interp - ti) / (tf - ti)
    
    
    # Speed is initially provided as km/s.
    unit_scale = ((tf-ti).to(u.s) * u.km / u.s).to(u.Mpc).value  
    
    sat_xi, sat_yi, sat_zi = np.asarray(sat_xi), np.asarray(sat_yi), np.asarray(sat_zi)
    sat_vxi, sat_vyi, sat_vzi = np.asarray(sat_vxi), np.asarray(sat_vyi), np.asarray(sat_vzi)
    sat_xf, sat_yf, sat_zf = np.asarray(sat_xf), np.asarray(sat_yf), np.asarray(sat_zf)
    sat_vxf, sat_vyf, sat_vzf = np.asarray(sat_vxf), np.asarray(sat_vyf), np.asarray(sat_vzf) 
    
    cen_xi, cen_yi, cen_zi = np.asarray(cen_xi), np.asarray(cen_yi), np.asarray(cen_zi)
    cen_vxi, cen_vyi, cen_vzi = np.asarray(cen_vxi), np.asarray(cen_vyi), np.asarray(cen_vzi)
    cen_xf, cen_yf, cen_zf = np.asarray(cen_xf), np.asarray(cen_yf), np.asarray(cen_zf)
    cen_vxf, cen_vyf, cen_vzf = np.asarray(cen_vxf), np.asarray(cen_vyf), np.asarray(cen_vzf)  
    
    #Find relative to central galaxy, and scale the units - velocities are now comoving.    
    xinit = np.array([sat_xi - cen_xi, sat_yi - cen_yi, sat_zi - cen_zi]).T / cosmo.h
    xfin  = np.array([sat_xf - cen_xf, sat_yf - cen_yf, sat_zf - cen_zf]).T / cosmo.h
    vinit = np.array([sat_vxi - cen_vxi, sat_vyi - cen_vyi, sat_vzi - cen_vzi]).T * unit_scale / cosmo.scale_factor(z_red_i)
    vfin  = np.array([sat_vxf - cen_vxf, sat_vyf - cen_vyf, sat_vzf - cen_vzf]).T * unit_scale / cosmo.scale_factor(z_red_f)
    
    def do_interpolation(xinit, xfin, tscaled):
        '''Writen in terms of position, but exactly the same algorithm for velocities'''
        xinit = np.atleast_2d(xinit)
        xfin = np.atleast_2d(xfin)
        tscaled = np.atleast_1d(tscaled)
        #print xinit.shape
        TOLERANCE = 1e-20
        
        # Find initial and final radii
        rinit = np.linalg.norm(np.atleast_2d(xinit), axis=1, keepdims=True)
        rfin  = np.linalg.norm(np.atleast_2d(xfin),  axis=1, keepdims=True)


        # Find coordinate system so that initial and final postions of the satalite galaxy are
        # in the XY plane, with centeral galaxy at origin.
        x_axis_vector = xinit / rinit
        z_axis_vector = np.cross(x_axis_vector, xfin)

        #For when the two position vectors are almost parrallel:
        where_parrallel = np.where(np.linalg.norm(z_axis_vector, axis=1)<TOLERANCE)[0]

        # In this case, produce a new vector known not to be parrallel to them, and use that
        # to produce the z_axis_vector
        if len(where_parrallel): 
            skewed_vec = np.random.rand(3)
            z_axis_vector[where_parrallel] = np.cross(x_axis_vector[where_parrallel], skewed_vec)
        
        z_axis_vector = z_axis_vector / np.linalg.norm(z_axis_vector,  axis=1, keepdims=True)
        
        # Get y axis vector
        y_axis_vector = np.cross(x_axis_vector, z_axis_vector, axis=1)
             
        # Get new x and y coordinates of position 2
        x_coords = np.einsum('ij,ij->i', x_axis_vector, xfin)
        y_coords = np.einsum('ij,ij->i', y_axis_vector, xfin)

        #get angles between them at the relevent times
        thetas = np.tensordot(tscaled, np.arctan2(y_coords,x_coords), axes=0)
        

        
        # Get new radii at times:
        radii = np.tensordot(tscaled, (rfin - rinit).flatten(), axes=0) + rinit.flatten()

        
        #Put back into origional coordinates.
        #I hate this bit of code, but it's the only way I can get it to work
      #  pos = np.zeros((len(times),len(rinit),3))
      #  for i in range(len(pos)):
      #      pos[i] = (np.multiply(x_coords.T, np.cos([i])) * radii[i] + np.multiply(y_coords.T, np.sin([i])) * radii[i]).T

        pos = np.einsum('ij,ki->kij', x_axis_vector, np.cos(thetas)*radii) + np.einsum('ij,ki->kij', y_axis_vector, np.sin(thetas)*radii)

        return pos
    
    #Put back into origional frame
    pos_interp = do_interpolation(xinit, xfin, tscaled)
    vel_interp = do_interpolation(vinit, vfin, tscaled)
    
    #return results
    #(times, (center position&velocity), (satalite position&velocity))
    #print cen_pos.shape
   # print pos_interp.shape
    pos_interp = np.broadcast_to(pos_interp.squeeze(), cen_pos.shape) * cosmo.h
    vel_interp = (np.broadcast_to(vel_interp.squeeze(), cen_vel.shape).T * cosmo.scale_factor(z_red_interpolate)).T/ unit_scale
    
    
    
    return times, (cen_pos, cen_vel), (cen_pos + pos_interp, cen_vel + vel_interp)   
    

def time_spaced_interpolation_sat_sat(sat_xi, sat_yi, sat_zi, sat_vxi, sat_vyi, sat_vzi,
                                      at_xf, sat_yf, sat_zf, sat_vxf, sat_vyf, sat_vzf,
                                      cen_xi, cen_yi, cen_zi, cen_vxi, cen_vyi, cen_vzi, 
                                      cen_xf, cen_yf, cen_zf, cen_vxf, cen_vyf, cen_vzf,
                                      z_red_i, z_red_f, 
                                      samples=50, return_redshifts=False):
    '''
    
    '''
    ti, tf = cosmo.age([z_red_i, z_red_f])
    ages = np.linspace(ti, tf, samples, endpoint=True)
    redshifts = np.zeros(samples)
    
    #Must be a better way to do this?
    for i, age in enumerate(ages):
        redshifts[i] = apcos.z_at_value(cosmo.age, age)
    
    times, pos, vel = interpolate_sat_sat(sat_xi, sat_yi, sat_zi, sat_vxi, sat_vyi, sat_vzi,
                                          at_xf, sat_yf, sat_zf, sat_vxf, sat_vyf, sat_vzf,
                                          cen_xi, cen_yi, cen_zi, cen_vxi, cen_vyi, cen_vzi,
                                          cen_xf, cen_yf, cen_zf, cen_vxf, cen_vyf, cen_vzf,
                                          z_red_i, z_red_f, redshifts)
    
    if return_redshifts:
        return times, pos, vel, redshifts
    else:
        return times, pos, vel
