# https://stackoverflow.com/questions/44318231/score-function-of-cox-proportional-hazard-in-jump
using JuMP, Ipopt

times = [6,7,10,15,19,25];
is_censored = 1 .- [1,0,1,1,0,1];
is_control = 1 .- [1,1,0,1,0,0];

uncensored = findall(is_censored .== 0)

println("times = $times")
println("is_censored = $is_censored")
println("is_control = $is_control")

#m = Model(solver=IpoptSolver(print_level=0))
m = Model(Ipopt.Optimizer)
set_optimizer_attributes(m, "tol" => 1e-6, "max_iter" => 100)
@variable(m, β, start = 0.0)
# The beginning of the expression inside the log uses a common trick of
# choosing a +1 or -1 sign according to a 0/1 value bool with (-1)^bool.
@NLobjective(m, Max, sum(log(1+(-1)^is_control[uncensored[i]]*
  sum((-1)^is_control[j]*exp(is_control[j]*β) for j=uncensored[i]:length(times))/
  sum( exp(is_control[j]*β) for j=uncensored[i]:length(times)))
     for i=1:length(uncensored)))

JuMP.optimize!(m)
println("β = ", JuMP.value(β))
