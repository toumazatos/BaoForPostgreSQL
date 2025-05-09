import torch
import warnings
warnings.filterwarnings("ignore", message=".*Using a target size .* that is different to the input size .*")
import torch.nn.functional as F
from TreeConvolution.util import prepare_trees
import os
import numpy as np
import torch.optim
import joblib
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import net
from featurize import TreeFeaturizer
from model import BaoRegression
import copy
import sqlite3
import json


CUDA = torch.cuda.is_available()

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_perts=False, random_start=True, n_restarts = 1, normalize = True):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.random_perturbations_only = random_perts
        self.random_start = random_start
        self.n_restarts = n_restarts
        self.loss_func = torch.nn.MSELoss()
        self.normalize = normalize

    
    def apply_gradient_perturbation(self, vectorized_tree, grad, epsilon, alpha):
        """
        Apply the gradients to perturb the vectorized tree structure.
        Gradients are directly mapped to their respective fields based on node indices.
        """
        def recursive_update(node, grad, node_idx):
            """
            Recursively apply perturbations to each node in the vectorized tree.
            """
            # Apply perturbation to the second-to-last and last elements of the node's feature array
            #print(f"isinstance(node, tuple) --> {isinstance(node, tuple)}")
            #print(f"len(node) > 0 --> {len(node) > 0}")
            if isinstance(node, tuple) and len(node) > 0:
                feature_vector = node[0]
                # Apply perturbation to the Total Cost (second-to-last element)
                cost_grad = grad[0, -2, node_idx].sign().item()
                feature_vector[-2] *= (1 + alpha * cost_grad)
                feature_vector[-2] = np.clip(
                    feature_vector[-2],
                    (1 - epsilon) * feature_vector[-2],
                    (1 + epsilon) * feature_vector[-2]
                )

                # Apply perturbation to the Plan Rows (last element)
                rows_grad = grad[0, -1, node_idx].sign().item()
                feature_vector[-1] *= (1 + alpha * rows_grad)
                feature_vector[-1] = np.clip(
                    feature_vector[-1],
                    (1 - epsilon) * feature_vector[-1],
                    (1 + epsilon) * feature_vector[-1]
                )

            # Increment the node index
            node_idx[0] += 1

            # Recursively update child nodes
            if isinstance(node, tuple) and len(node) > 1:
                left = left_child(node)
                right = right_child(node)
                if left:
                    recursive_update(left, grad, node_idx)
                if right:
                    recursive_update(right, grad, node_idx)

        # Initialize node index as a mutable list to maintain state across recursive calls
        node_idx = [0]

        try:
            recursive_update(vectorized_tree[0], grad, node_idx)
        except IndexError as e:
            print(f"Error during perturbation: {str(e)}")

        return vectorized_tree

    
    def perturb_input_random(self, tree):
        """Recursively applies a perturbation to 'Total Cost' and 'Plan Rows' in the tree dictionary."""
        if "Total Cost" in tree:
            pert = tree["Total Cost"] * (1 + np.random.uniform(-1, 1) * self.epsilon)
            if pert < 0:  # Ensure 'Total Cost' doesn't fall below zero
                pert = 1e-6
            tree["Total Cost"] = pert # apply random perturbation 
        
        if "Plan Rows" in tree:
            pert = tree["Plan Rows"] * (1 + np.random.uniform(-1, 1) * self.epsilon)
            if pert < 0:  # Ensure 'Plan Rows' doesn't fall below zero
                pert = 1e-6
            tree["Plan Rows"] = pert # apply random perturbation 
        
        # Recursively apply perturbations to nested 'Plans', if they exist
        if "Plans" in tree:
            for i in range(len(tree["Plans"])):
                self.perturb_input_random(tree["Plans"][i])  # Recursive call
        
        # Ensure the "Buffers" field is passed down to the perturbed plan
        if "Buffers" in tree:
            tree["Buffers"] = tree["Buffers"]
        
        return tree

    
    def perturb_input(self, tree, grad, epsilon, alpha):
        """Perturb the tree using gradients."""
        # perturbed_tree = copy.deepcopy(tree)
        return self.apply_gradient_perturbation(tree, grad, epsilon, alpha)
    

    def perturb(self, x_nat, y): # mostly used for random perturbations
        """Generate adversarial examples within epsilon of x_nat in l_infinity norm."""        
        x = copy.deepcopy(x_nat)

        if self.random_perturbations_only: # add random perturbations
            # Apply random perturbation to the original input data 
            perturbed_plan = self.perturb_input_random(x['Plan'])
            x['Plan'] = perturbed_plan
        return x

    

    def perturb_v1_vector(self, x_nat, y): # The correct implementation for PGD
        """Generate adversarial examples using the model's gradients."""
        
        x_adv = self.model.featurize_vector(x_nat)
        x_clean = self.model.featurize_vector(x_nat) 

        grad, _ = self.model.compute_gradients_vectorized(x_adv, y, layer=0, return_loss=True) # needed to apply fit()
        clean_grad = grad
        #print_grad = True
        """print("Before")
        print(f'x_adv = {x_adv}')
        print(f'grad = {grad}')"""
        

        for step in range(self.k):
            # ---- Compute gradients ----
            
            # objective function is MSLoss
            grad, loss = self.model.compute_gradients_vectorized(x_adv, y, layer=0, return_loss=True) 

            # objective function is model prediction (query cost)
            #grad, loss_grad = self.model.compute_gradients_vectorized_output(x_adv, y, layer=0) 

            #print(f'grad = {grad}')
            #print(f'shape = {grad.shape}')
            
            """if step == 0: 
                print(f'grad wrt output = {grad}')
                print("\n")
                print(f'grad wrt loss = {loss_grad}')"""

            # Apply perturbation based on gradients
            x_adv = self.perturb_input(x_adv, grad, self.epsilon, self.a) #only in vector space
            

            # Debugging: Print loss and gradient info
            if self.normalize:
                x_adv = [normalize_tree_bottom_up(node) for node in x_adv]
            x_adv = scale_cost_values(x_adv)

        return (self.model.predict_vector(x_adv), x_adv, x_clean, clean_grad, grad) # return (predicted_time, adv_vector, clean_vector)



# Normalization functions
def normalize_tree_top_down(tree, parent_cost=float('inf')):
    """
    Recursively normalize the tree so that each node's cost (second-to-last element in its array)
    is never greater than its parent node's cost.
    """
    if isinstance(tree, tuple) and isinstance(tree[0], np.ndarray):  # If it's a node with children or leaf node
        node = np.copy(tree[0])  # Copy to avoid modifying original structure
        node_cost = node[-2]

        # Ensure node cost does not exceed parent cost
        node[-2] = min(node_cost, parent_cost)

        # Process children if they exist
        normalized_children = tuple(normalize_tree_top_down(child, node[-2]) for child in tree[1:])
        
        return (node, *normalized_children)
    
    return tree  # Return unchanged if not a valid node structure


def normalize_tree_bottom_up(tree):
    """
    Recursively normalize the tree so that each node's cost (second-to-last element in its array)
    is at least as high as the maximum of its children's costs.
    """
    if isinstance(tree, tuple):  # If it's a node with children
        node, *children = tree
        node = np.array(node)  # Ensure it's a NumPy array to modify it
        
        # Recursively normalize children first
        normalized_children = [normalize_tree_bottom_up(child) for child in children]
        
        # Get the maximum cost from children
        max_child_cost = max((child[0][-2] for child in normalized_children if isinstance(child, tuple)), default=node[-2])
        
        # Ensure node cost is at least as high as the maximum child cost
        node[-2] = max(node[-2], max_child_cost)
        
        return (node, *normalized_children)
    else:  # If it's a leaf node
        return tree

def scale_cost_values(tree, max_cost=None):
    """
    Scale all cost values in the tree to be within the [0, 1] range.
    """
    if max_cost is None:
        # Find the maximum cost value in the tree
        all_costs = []
        def collect_costs(subtree):
            if isinstance(subtree, tuple) and isinstance(subtree[0], np.ndarray):
                all_costs.append(subtree[0][-2])
                for child in subtree[1:]:
                    collect_costs(child)
        for node in tree:
            collect_costs(node)
        max_cost = max(all_costs) if all_costs else 1.0
    
    # Normalize costs
    def scale_subtree(subtree):
        if isinstance(subtree, tuple) and isinstance(subtree[0], np.ndarray):
            node = np.copy(subtree[0])  # Copy to avoid modifying the original
            node[-2] /= max_cost  # Scale cost
            scaled_children = tuple(scale_subtree(child) for child in subtree[1:])
            return (node, *scaled_children)
        return subtree
    
    return [scale_subtree(node) for node in tree]



# Data Loader
def fetch_all_experience(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT plan, reward, arm_idx, query_name FROM experience WHERE id > 25 AND id <= 500;")
    results = c.fetchall()
    conn.close()
    return (results)



def run_experiment(epsilon, k, a, normalize=True):
    # Example parameters
    epsilon = epsilon
    k = k
    a = a
    random = False
    
    reg = BaoRegression(have_cache_data=True)
    reg.load("../models/default_model/")

    bao_model = reg  # Set in_channels to the appropriate value
    
    # Initialize the attack
    # TODO: try more random starts
    grad_attack = LinfPGDAttack(bao_model, epsilon, k, a, random, n_restarts=1, normalize=normalize)
    random_attack = LinfPGDAttack(bao_model, epsilon, k, a, random_perts=True)

    db_path = '../db_snapshots/adv/clean/bao_snapshot_with_6_arms.db'
    #db_path = '../db_snapshots/500/bao_snapshot_with_6_arms.db'
    data_ = fetch_all_experience(db_path)
    #clean_preds = [d[1] for d in data]
    pg_times = []
    print("data_[0] = {}".format(data_[0][0]))
    print("data_[1] = {}".format(data_[0][1]))
    print("data_[2] = {}".format(data_[0][2]))
    print("data_[3] = {}".format(data_[0][3]))
    for item in data_:
        pg_times.append(item[1])
    
    # data_ = data_[:1] 
    arms = [] # get arms from experience (for studying worst cases)
    for item in data_:
        arms.append(item[2])
    
    query_names = [] # get query_names from experience (for studying worst cases)
    for item in data_:
        query_names.append(item[3])


    clean_preds = []
    for item in data_:
        clean_preds.append(reg.predict(json.loads(item[0])))
    
    data = []
    for i in range(len(data_)):
        data.append((data_[i][0], clean_preds[i]))
     

    print("# =============== Original attack =================")
    adv_preds = []
    adv_vectors = [] #vectors after pgd
    clean_vectors = [] # vectors before pgd
    clean_plans = [] # 
    adv_grads = []
    clean_grads = []
    # Generate adversarial examples (PGD)
    for item in data: 
        reg.predict(json.loads(item[0]))
        clean_plans.append(json.loads(item[0]))
        #print("New example")
        x_adv = grad_attack.perturb_v1_vector(json.loads(item[0]), item[1])

        adv_preds.append(x_adv[0])
        adv_vectors.append(x_adv[1])
        clean_vectors.append(x_adv[2])
        clean_grads.append(x_adv[3])
        adv_grads.append(x_adv[4])

    #print(adv_preds)
    #record_examples("../db_snapshots/adv/examples/e0_{}.db".format(str(epsilon).split(".")[1]), adv_preds) # record examples


    random_preds = []
    for item in data: 
        x_rand = random_attack.perturb(json.loads(item[0]), item[1])
        #x_rand['Plan'] = round_plan_rows(x_rand['Plan'])
        pred = reg.predict(x_rand)
        random_preds.append(pred)

    deltas = []
    deltas_alt = [] # TODO: remove
    for i in range(len(clean_preds)):
        deltas_alt.append(abs(clean_preds[i] - adv_preds[i])/ clean_preds[i])
        deltas.append((adv_preds[i] - clean_preds[i])/ clean_preds[i])
    deltas_ = np.array(deltas.copy())
    pg_times_ = np.array(pg_times.copy())

    
    # ==================== PGD ATTACK =======================
    deltas = np.array([deltas])
    print("Overall:")
    print("Mean: {:.2f}".format(np.mean(deltas)))
    print("Min: {:.2f}".format(np.min(deltas)))
    print("Max: {:.2f}".format(np.max(deltas)))
    count_greater_than_epsilon = np.sum(deltas > epsilon)
    percentage_greater_than_epsilon = (count_greater_than_epsilon / len(np.array(deltas).flatten() )) * 100
    print("Prct of deltas greater than epsilon: {:.2f}%".format(percentage_greater_than_epsilon))
    print("--------------------- Old Deltas ---------------------") # TODO: remove
    print("Overall:")
    print("Mean: {:.2f}".format(np.mean(deltas_alt)))
    print("Min: {:.2f}".format(np.min(deltas_alt)))
    print("Max: {:.2f}".format(np.max(deltas_alt)))
    print("--------------------- Old Deltas ---------------------") # TODO: remove

    sorted_indices = np.argsort(-pg_times_)  # Sorting in descending order
    pg_times_sorted = pg_times_[sorted_indices]
    deltas_sorted = np.array(deltas_)[sorted_indices]
    query_names_sorted = np.array(query_names)[sorted_indices]
    arms_sorted = np.array(arms)[sorted_indices]
    """
    adv_vectors_sorted = [adv_vectors[i] for i in sorted_indices]
    clean_vectors_sorted = [clean_vectors[i] for i in sorted_indices]
    clean_plans_sorted = [clean_plans[i] for i in sorted_indices]
    adv_grads_sorted = [adv_grads[i] for i in sorted_indices]
    clean_grads_sorted = [clean_grads[i] for i in sorted_indices]
    """

    

    print("50 longest Queries:")
    print("Mean: {:.2f}".format(np.mean(deltas_sorted[-50:])))
    print("Min: {:.2f}".format(np.min(deltas_sorted[-50:])))
    print("Max: {:.2f}".format(np.max(deltas_sorted[-50:])))
    #deltas_.sort(reverse=True)
    sorted_indices = np.argsort(-deltas_)  # Sorting in descending order
    deltas_sorted = deltas_[sorted_indices]
    pg_times_sorted = np.array(pg_times_)[sorted_indices]

    print("adv = {}".format(deltas_sorted.tolist()))
    print("\n")
    #print("pg_times = {}".format(pg_times_sorted.tolist()))

    print("Worst Case Queries & Arms (10)")
    print("queries = {}".format(query_names_sorted[-10:]))
    print("arms = {}".format(arms_sorted[-10:]))

    '''
    print(f" BEFORE: \n {clean_vectors_sorted[0]}")
  
    print(f" AFTER: \n {adv_vectors_sorted[0]}")
    '''

    



    #print("Adversarial Examples:", x_adv)
    # ==================== RANDOM ATTACK =======================
    print("Random:")
    deltas_rand = []
    for i in range(len(clean_preds)):
        deltas_rand.append(abs(clean_preds[i] - random_preds[i])/ clean_preds[i])
    deltas_rand_ = deltas_rand.copy()
    deltas_rand = np.array([deltas_rand])
    print("Overall:")
    print("Mean: {:.2f}".format(np.mean(deltas_rand)))
    print("Min: {:.2f}".format(np.min(deltas_rand)))
    print("Max: {:.2f}".format(np.max(deltas_rand)))
    count_greater_than_epsilon = np.sum(deltas_rand > epsilon)
    percentage_greater_than_epsilon = (count_greater_than_epsilon / len(deltas_rand)) * 100
    print("Prct of deltas greater than epsilon: {:.2f}%".format(percentage_greater_than_epsilon))
    deltas_rand_.sort(reverse=True)
    print("rand = {}".format(deltas_rand_))
    """deltas_rand_ = np.array(deltas_rand_)
    deltas_rand_ = deltas_rand_[sorted_indices]
    print("rand = {}".format(deltas_rand_.tolist()))"""

    
    
if __name__ == '__main__':

    print("============================E 0.5 clipped (PGD) =================================")
    #run_experiment(epsilon=0.5, k=40, a= 0.025) # FGSM
    print("\n")
    run_experiment(epsilon=0.5, k=40, a= 0.025, normalize=True) # FGSM // TODO: try more random starts
    print("\n")