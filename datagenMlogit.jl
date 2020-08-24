using Random
function datagenMlogit()
	## Generate data for a linear model to test optimization
	#srand(1234)
	Random.seed!(1234);

	N = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!
	T = 5
	global J = 5
	global n = convert(Int64,N*T) # use -const- as a way to declare a variable to be global (so other functions can access it)

	# generate the covariates
	X = [ones(N*T,1) 5 .+ 3*randn(N*T,1) rand(N*T,1) 2.5 .+ 2*randn(N*T,1)]
	global K1 = size(X,2)
	Z = zeros(N*T,3,J)
	for j=1:J
		Z[:,:,j] = [3 .+ randn(N*T,1) randn(N*T,1) .- 1 rand(N*T,1)];
	end
	global K2 = size(Z,2)
	global baseAlt = 5

	# X coefficients
	bxAns      = zeros(size(X,2),J)
	bxAns[:,1] = [-0.15 0.10  0.50 0.10 ]
	bxAns[:,2] = [-1.50 0.15  0.70 0.20 ]
	bxAns[:,3] = [-0.75 0.25 -0.40 0.30 ]
	bxAns[:,4] = [ 0.65 0.05 -0.30 0.40 ]
	bxAns[:,5] = [ 0.75 0.10 -0.50 0.50 ]
	# bxAns[:,5] = [ 0.0  0.0   0.00 0.00 ]

	# Z coefficients
	bzAns = [.2;.5;.8];

	# generate choice probabilities
	u   = zeros(N*T,J)
	p   = zeros(N*T,J)
	dem = zeros(N*T,1)
	for j=1:J
		u[:,j] = X*bxAns[:,j]+Z[:,:,j]*bzAns
		dem[:] += exp.(u[:,j])
	end
	for j=1:J
		p[:,j] = exp.(u[:,j])./dem
	end

	# use the choice probabilities to create the observed choices
	draw=rand(N*T,1)
	Y=(draw.<sum(p[:,1:end],dims=2))+
	  (draw.<sum(p[:,2:end],dims=2))+
	  (draw.<sum(p[:,3:end],dims=2))+
	  (draw.<sum(p[:,4:end],dims=2))+
	  (draw.<sum(p[:,5:end],dims=2))

	# return generated data so that other functions (below) have access
	return X,Y,Z,bxAns,bzAns,n,J,K1,K2,baseAlt
end
