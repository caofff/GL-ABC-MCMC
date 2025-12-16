import torch
import os
import normflows as nf


class MCMCRunner:
    def __init__(self, abc_set, output_dir='./'):
        """
        Args:
            abc_set: ABC set for the problem
            output_dir: Directory to save results
        """
        self.abc_set = abc_set
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_global_mcmc(self, num_iterations,initial_theta, initial_y,global_frequency, local_proposal, global_proposal,
                    output_file='global_mcmc_results.csv'):
        """Run Global MCMC method"""
        from .GlobalMCMC import GlobalMCMC
        if output_file is not None:
            output_file = os.path.join(self.output_dir, output_file)

        return GlobalMCMC(
            ABCset=self.abc_set,
            num_ite=num_iterations,
            Initial_theta=initial_theta,
            Initial_y=initial_y,
            Global_Proposal= global_proposal,
            filelocation=output_file,
            global_frequency=global_frequency,
            Local_Proposal=local_proposal
        )

    def run_glmcmc(self, num_iterations, initial_theta, initial_y, global_frequency, local_proposal, importance_proposal,
                   batch_size, output_file='glmcmc_results.csv'):
        """Run GLMCMC method"""
        from .GLMCMC import GLMCMC

        if output_file is not None:
            output_file = os.path.join(self.output_dir, output_file)

        return GLMCMC(
            ABCset=self.abc_set,
            num_ite=num_iterations,
            Initial_theta=initial_theta,
            Initial_y=initial_y,
            Local_Proposal=local_proposal,
            filelocation=output_file,
            global_frequency=global_frequency,
            Importance_Proposal=importance_proposal,
            batch_size=batch_size
        )

    def run_aglmcmc(self, num_iterations, initial_theta, initial_y, global_frequency, local_proposal, Initial_ISIR_prop,
                      batch_size, step_size, alpha, hat_eps_T, output_file='glmcmc_results.csv'):
        """Run AGLMCMC method"""
        from .AGLMCMC import AGLMCMC

        if output_file is not None:
            output_file = os.path.join(self.output_dir, output_file)

        return AGLMCMC(
            ABCset=self.abc_set,
            num_ite=num_iterations,
            Initial_theta=initial_theta,
            Initial_y=initial_y,
            Local_Proposal=local_proposal,
            Initial_ISIR_prop= Initial_ISIR_prop,
            filelocation=output_file,
            global_frequency=global_frequency,
            step_size=step_size,
            batch_size=batch_size,
            alpha = alpha,
            hat_eps_T =hat_eps_T
        )

    def run_glmala(self, num_iterations, initial_theta, initial_y, global_frequency, importance_proposal,
                   batch_size, tau, num_grad,
                   output_file='glmala_results.csv'):
        """Run GLMALA method"""
        from .GLMALA import GLMALA

        if output_file is not None:
            output_file = os.path.join(self.output_dir, output_file)

        return GLMALA(
            ABCset=self.abc_set,
            num_ite=num_iterations,
            Initial_theta=initial_theta,
            Initial_y=initial_y,
            tau=tau,
            num_grad=num_grad,
            filelocation=output_file,
            global_frequency=global_frequency,
            Importance_Proposal=importance_proposal,
            batch_size=batch_size
        )

    def run_glmcmc_nf(self, num_iterations, initial_theta, initial_y, global_frequency, local_proposal, importance_proposal_base,batch_size,
                      step_size, train_steps,
                      output_file='glmcmc_nf_results.csv'):
        """Run GLMCMC with Normalizing Flows"""
        from .GLMCMC_NFs import GLMCMC_NF

        if output_file is not None:
            output_file = os.path.join(self.output_dir, output_file)

        return GLMCMC_NF(
            ABCset=self.abc_set,
            num_ite=num_iterations,
            Initial_theta=initial_theta,
            Initial_y=initial_y,
            Local_Proposal=local_proposal,
            filelocation=output_file,
            global_frequency=global_frequency,
            step_size=step_size,
            batch_size=batch_size,
            base=importance_proposal_base,
            Train_step=train_steps
        )