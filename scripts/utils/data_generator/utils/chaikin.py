def Sum_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return x1+x2, y1+y2

def Multiply_point(multiplier, P):
    x, y = P
    return float(x)*float(multiplier), float(y)*float(multiplier)

def Check_if_object_is_polygon(Cartesian_coords_list):
    if Cartesian_coords_list[0] == Cartesian_coords_list[len(Cartesian_coords_list)-1]:
        return True
    else:
        return False

def Find_Q_point_position(P1, P2):
    Summand1 = Multiply_point(float(3)/float(4), P1)
    Summand2 = Multiply_point(float(1)/float(4), P2)
    Q = Sum_points(Summand1, Summand2) 
    return Q

def Find_R_point_position(P1, P2):
    Summand1 = Multiply_point(float(1)/float(4), P1)
    Summand2 = Multiply_point(float(3)/float(4), P2)        
    R = Sum_points(Summand1, Summand2)
    return R

def Smooth_by_Chaikin(Cartesian_coords_list, number_of_refinements):
    refinement = 1
    copy_first_coord = Check_if_object_is_polygon(Cartesian_coords_list)
    while refinement <= number_of_refinements:
        New_cartesian_coords_list = []

        for num, tuple in enumerate(Cartesian_coords_list):
            if num+1 == len(Cartesian_coords_list):
                pass
            else:
                P1, P2 = (tuple, Cartesian_coords_list[num+1])
                Q = Find_Q_point_position(P1, P2)
                R = Find_R_point_position(P1, P2)
                New_cartesian_coords_list.append(Q)
                New_cartesian_coords_list.append(R)

        if copy_first_coord:
            New_cartesian_coords_list.append(New_cartesian_coords_list[0])

        Cartesian_coords_list = New_cartesian_coords_list
        refinement += 1
    return Cartesian_coords_list