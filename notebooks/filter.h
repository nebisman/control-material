float computeController(float e){
    // This function computes the control law for a discretized controller Cd(z)
    // e = ref - y is the current error in the controlled system


    // The following constants define the A matrix
    const float a1_1 = 0.80180180072784423828125000000000;
    const float a1_2 = 0.00900900922715663909912109375000;

    // The following constants define the B matrix
    const float b1 = -0.05656756833195686340332031250000;
    const float b2 = 0.04199999943375587463378906250000;

    // The following constants define the C matrix
    const float c1 = 0.90090090036392211914062500000000;
    const float c2 = 0.00450450461357831954956054687500;

    // The following constant define the D scalar
    const float d = 0.37921622395515441894531250000000;

    // The following variables represent the states x[n]
    // in the state-space representation. They must be declared
    // as static to retain their values between function calls.
    static float x1 = 0;
    static float x2 = 0;

    // The following variables are the new computed states x[n+1]
    // of the state space representation
    float x1_new = 0;
    float x2_new = 0;

    // The following variable is control signal u  
    // it also must be declared as static to retain its value between function calls. 
    float u = 0;

    /*************************************************
                THIS IS THE CONTROLLER'S CODE
    **************************************************/

    // Compute the new state x[n+1] = A*x[n] + B*e[n]
    x1_new = a1_1*x1 + a1_2*x2 + b1*e;
    x2_new = x2 + b2*e;

    // Compute the control output u[n] = C*x[n] + D*e[n]
    u = c1*x1 + c2*x2 + d*e;

    // Make the next state the current state: x[n] <- x[n+1]
    x1 = x1_new;
    x2 = x2_new;

    // now, the filtered signal is available to the main routine
    return u;
}
