function jumpMlogitMLEnoRestr()
	## MLE of multinomial logit model
	# Declare the name of your model and the optimizer you will use
	# myMLE = Model(solver=IpoptSolver(tol=1e-6))
	myMLE = Model(Ipopt.Optimizer)
	set_optimizer_attributes(myMLE, "tol" => 1e-6, "max_iter" => 100)

	# Declare the variables you are optimizing over
	@variable(myMLE, bx[k=1:K1,j=1:J-1])
	# @variable(myMLE, bx[k=1:K1,j=setdiff(1:J,baseAlt)])
	@variable(myMLE, bz[k=1:K2])

	# # Write constraints here if desired (in this case, set betas for base alternative to be 0)
	# for k=1:K1
	  # @constraint(myMLE, bx[k,baseAlt] == 0)
	# end

	# Write the objective function
	# likelihood = @expression(myMLE, sum( sum( (Y[i].==j)*( sum( X[i,k]*(bx[k,j]) for k=1:K1) + sum( (Z[i,k,j]-Z[i,k,baseAlt])*bz[k] for k=1:K2) ) for j=1:J if j!=baseAlt) - log.( 1 + sum( exp.( sum( X[i,k]*(bx[k,j]) for k=1:K1) + sum( (Z[i,k,j]-Z[i,k,baseAlt])*bz[k] for k=1:K2)  ) for j=1:J if j!=baseAlt) ) for i=1:n));

	# Set the objective function to be maximized
	# @objective(myMLE, Max, likelihood )

	# combine expression and objective as @NLobjective
	@NLobjective(myMLE,Max, sum(
								sum( (Y[i]==j)*( sum( X[i,k]*(bx[k,j]) for k=1:K1) +
								sum( (Z[i,k,j]-Z[i,k,baseAlt])*bz[k] for k=1:K2) ) for j=1:J if j!=baseAlt) -
								log( 1 + sum( exp( sum( X[i,k]*(bx[k,j]) for k=1:K1) +
								sum( (Z[i,k,j]-Z[i,k,baseAlt])*bz[k] for k=1:K2)  ) for j=1:J if j!=baseAlt) )
							for i=1:n)
				)

	# Solve the objective function
	#status = solve(myMLE)
	status = JuMP.optimize!(myMLE)

	# Generate Hessian
	this_par = myMLE.colVal
	m_const_mat = JuMP.prepConstrMatrix(myMLE)
	m_eval = JuMP.JuMPNLPEvaluator(myMLE, m_const_mat);
	MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
	hess_struct = MathProgBase.hesslag_structure(m_eval)
	hess_vec = zeros(length(hess_struct[1]))
	numconstr = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
	dimension = length(myMLE.colVal)
	MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
	this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
	# println(full(this_hess_ld))
	hOpt = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
	hOpt = -full(hOpt) #since we are maximizing
	# println(full(this_hess))

	# Calculate standard errors
	seOpt = sqrt(diag(hOpt\eye(size(hOpt,1))))
	# swap indices because order of coefficients is different than ordering of Hessian rows
	# note: it would be better to do this re-ordering to the Hessian, but I can't figure out...
	# how to do this simultaneously for rows and columns
	ind = [1:size(hOpt,1)]
	for j=1:J-1
		ind[(j-1)*K1+1:j*K1] = [j:K1:(J-1)*K1]
	end
	seOpt = seOpt[ind]

	# Save estimates
	bxOpt = JuMP.value(bx[:,:])
	bzOpt = JuMP.value(bz[:])
	# hOpt  = -eye(length(bx)+length(bz))

	# Print estimates and log likelihood value
	println("beta = ", bxOpt[:,:])
	println("bz = ", bzOpt[:])
	println("MLE objective: ", JuMP.objective_value(myMLE))
	println("MLE status: ", status)
	return bxOpt,bzOpt,hOpt,seOpt
end
