import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import argparse

max_car = 20
max_move_car =5
rental_first_lamada =3
rental_second_lamada=4
return_first_lamada =3
return_second_lamada=2
discount=0.9
rental_credit=10
move_car_cost=2
actions = np.linspace(-max_move_car,max_move_car,11,dtype=int)
print(actions)

possion_upper_bound =11
poisson_cache = dict()
def poisson_pro(n,lam):
    global poisson_cache
    key = 10*n + lam #give each possion sample a name
    if key not in poisson_cache:
        poisson_cache[key]=poisson.pmf(n,lam)
    return poisson_cache[key]


def one_iteration(state,action,state_value,constant_return_cars):
    #input one state and action and current_state-values, it outputs the returns for the q(s,a) by using bellman equation
    #q(s,a) = p(s'|s,a)*gamma*v(s') + r(s,a)*reward
    if action>=1:
        action1 = action - 1
    else:
        action1 = action
    reward1 = -abs(action1)*move_car_cost

    NUM_car_first = state[0] -action
    NUM_car_second = state[1] + action

    if NUM_car_first >10:
        reward1 = reward1 - 4
    if NUM_car_second >10:
        reward1 = reward1 -4

    returns = reward1

    for rental_first in range(possion_upper_bound):
        for rental_second in range(possion_upper_bound):
            prob_rental = poisson_pro(rental_first,rental_first_lamada)*poisson_pro(rental_second,rental_second_lamada)
            number_car_first = NUM_car_first
            number_car_second = NUM_car_second
            true_rental_first = min(rental_first,number_car_first)
            ture_rental_second = min(number_car_second,rental_second)

            reward2 = (true_rental_first+ture_rental_second)*rental_credit

            number_car_first -= true_rental_first
            number_car_second -= ture_rental_second

            if constant_return_cars:
                returned_first = return_first_lamada
                returned_second = return_second_lamada

                number_car_first = int(min((returned_first + number_car_first), max_car))
                number_car_second = int(min((returned_second + number_car_second), max_car))
                # print(discount,state_value,number_car_second_loc,number_car_first_loc)
                #returns += prob_rental*(reward2+discount * state_value[number_car_first, number_car_second])
                returns += prob_rental * (reward2+discount * state_value[number_car_first, number_car_second])
            else:
                for returned_first in range(possion_upper_bound):
                    for returned_second in range(possion_upper_bound):
                        prob_return = poisson_pro(returned_first,return_first_lamada)*poisson_pro(returned_second,return_second_lamada)
                        number_car_first = int(min(returned_first+number_car_first, max_car))
                        number_car_second = int(min(returned_second+number_car_second, max_car))

                        pro = prob_return*prob_rental
                        #print(number_car_first,number_car_second)
                        #print(pro,discount,state_value)
                        returns += pro*(reward2+discount*state_value[number_car_first,number_car_second])
    return returns

def policy_iteration(args):
    state_values = np.zeros((max_car+1,max_car+1))
    policy = np.zeros(state_values.shape)
    iterations = 0
    _,axes = plt.subplots(2,3,figsize=(20,10))
    #plt.subplots_adjust(wspace=0.1,hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=10)
        fig.set_yticks(list(reversed(range(max_car + 1))))
        fig.set_xlabel('# cars at second location', fontsize=10)
        fig.set_title('policy {}'.format(iterations), fontsize=10)
    #evaluation
        while True:
            old_state_values = state_values.copy()
            for i in range(max_car+1):
                for j in range(max_car+1):
                    new_state_value = one_iteration([i,j],policy[i,j],state_values,args.constant_return)
                    state_values[i,j] = new_state_value
            difference = abs(old_state_values-state_values).max()
            print('max value change{}'.format(difference))
            if difference < 1e-3:
                break

        #policy improvement
        policy_old = policy.copy()
        for i in range(max_car+1):
            for j in range(max_car+1):
                action_list = []
                for a in actions:
                    #print(a)
                    if (0<=a<=i) or (-j<=a<=0):
                        action_list.append(one_iteration([i,j],a,state_values,args.constant_return))
                    else:
                        action_list.append(-np.inf)
                best = actions[np.argmax(action_list)]
                policy[i,j]=best
        policy_stable=np.array_equal(policy_old,policy)
        print('policy stable {}'.format(policy_stable))
        if policy_stable:
            print('policy stable')
            fig = sns.heatmap(np.flipud(state_values), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=10)
            fig.set_yticks(list(reversed(range(max_car + 1))))
            fig.set_xlabel('# cars at second location', fontsize=10)
            fig.set_title('optimal value', fontsize=10)
            break

        iterations +=1
        print(iterations)

    plt.savefig('jack_b')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--constant_return', action='store_true')
    args = parser.parse_args()
    policy_iteration(args)






