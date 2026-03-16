rule CWE416_UseAfterFree
{
    meta:
        cwe         = "CWE-416"
        severity    = "HIGH"
        description = "Memory used after being freed"

    strings:
        $free  = "free(" ascii
        $use1  = "->" ascii
        $use2  = "= *" ascii
        $safe1 = "= NULL" ascii

    condition:
        $free and any of ($use*) and not $safe1
}

rule CWE416_UseAfterFree_Reassign
{
    meta:
        cwe         = "CWE-416"
        severity    = "HIGH"
        description = "Freed pointer not set to NULL before reuse"

    strings:
        $free   = "free(" ascii
        $malloc = "malloc(" ascii
        $null   = "= NULL" ascii

    condition:
        $free and $malloc and not $null
}