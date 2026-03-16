rule CWEOther_RaceCondition
{
    meta:
        cwe         = "CWE-Other"
        severity    = "MEDIUM"
        description = "Potential race condition with shared resources"

    strings:
        $thread1 = "pthread_create(" ascii
        $thread2 = "CreateThread(" ascii
        $shared1 = "global" ascii
        $shared2 = "static " ascii
        $mutex1  = "pthread_mutex_lock(" ascii
        $mutex2  = "EnterCriticalSection(" ascii

    condition:
        any of ($thread*) and
        any of ($shared*) and
        not any of ($mutex*)
}

rule CWEOther_HardcodedCredentials
{
    meta:
        cwe         = "CWE-Other"
        severity    = "HIGH"
        description = "Hardcoded passwords or credentials in code"

    strings:
        $pass1  = "password" nocase ascii
        $pass2  = "passwd" nocase ascii
        $pass3  = "secret" nocase ascii
        $pass4  = "api_key" nocase ascii
        $assign = "= \"" ascii

    condition:
        any of ($pass*) and $assign
}