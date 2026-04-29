import sys
sys.path.insert(0, '.')
from server.environment import IncidentResponseEnv

env = IncidentResponseEnv()

# Test 1: well-formed belief with quoted keys
t1 = 'Thought: auth logs show errors\nBelief: {"auth-service": 0.7, "db-primary": 0.2, "cache-cluster": 0.1}'
r1 = env._parse_belief_from_reasoning(t1)
print('Test 1:', r1)
assert r1 == {'auth-service': 0.7, 'db-primary': 0.2, 'cache-cluster': 0.1}, f'FAIL: {r1}'

# Test 2: no belief pattern → None
t2 = 'I think it is auth-service because of the errors'
r2 = env._parse_belief_from_reasoning(t2)
print('Test 2:', r2)
assert r2 is None, f'FAIL: expected None, got {r2}'

# Test 3: unquoted hyphenated keys
t3 = 'Belief: {auth-service: 0.6, db-primary: 0.3, cache-cluster: 0.1}'
r3 = env._parse_belief_from_reasoning(t3)
print('Test 3:', r3)
assert r3 == {'auth-service': 0.6, 'db-primary': 0.3, 'cache-cluster': 0.1}, f'FAIL: {r3}'

print('All 3 parse tests PASS')
