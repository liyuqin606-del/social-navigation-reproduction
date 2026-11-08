# TACC current wall-time limits — 2026-11-03

Scope: read-only SSH scheduler/account queries and public official documentation/source review. No job submitted, no allocation requested, no scheduler or account configuration changed.

## Conclusion

As observed at 2026-11-03 08:02–08:06 UTC (16:02–16:06 China time), this account's current native Slurm route has a **one-hour per-job partition limit**. This is not a default-time misunderstanding. The 2025 SING paper indeed describes a **24-hour peak-period operational policy**; that historical service policy and this live scheduler configuration differ. We have not verified a current 24-hour route for this account.

## Live evidence

- `sinfo --all --format=%P|%l|%L|%a` returns only `tacc*|1:00:00|n/a|up`.
- `scontrol show partition`: `PartitionName=tacc`, `MaxTime=01:00:00`, `DefaultTime=NONE`, `QoS=N/A`, `AllowQos=ALL`, `MaxNodes=2`.
- The account association is `tacc|tacc|liyuqin||taccqos||`: only `taccqos` is listed for this user; no association MaxWall is set.
- `taccqos`: no MaxWall and no Flags; MaxJobsPU=2. An unset QOS MaxWall does not remove the partition MaxTime. No observed QOS has the PartitionTimeLimit override flag.
- `AccountingStorageEnforce=associations,limits,qos`; `JobSubmitPlugins=(null)`; scheduler `sched/backfill`; Slurm 17.11.2.
- `sbatch` resolves to `/usr/bin/sbatch`, an ELF executable, not a visible shell wrapper.

Raw query outputs are in [first probe](tacc_time_limits_remote_20261103.json) and [follow-up](tacc_time_limits_remote_followup_20261103.json). The assoc_mgr command also returns scheduler accounting state beyond the selected user; only the user's allowed QOS and relevant limit fields are used in this conclusion.

[Slurm sinfo documentation](https://slurm.schedmd.com/sinfo.html) defines `%l` as maximum time and `%L` as default time. [Slurm QOS documentation](https://slurm.schedmd.com/qos.html) explains that PartitionTimeLimit allows a QOS to override a partition time limit; no such flag was returned here.

## What the 24-hour evidence actually says

The authors' [ASPLOS 2025 paper](https://home.cse.ust.hk/~kaichen/papers/tacc-asplos25.pdf), section 5.3.1, PDF page 13, describes limiting jobs to 24 hours during peak periods, with peak periods exemplified by another job waiting over an hour. It recommends checkpointing/resubmission for longer work. It does not establish the current user's 2026 scheduler entitlement.

The [current TACC site](https://tacc.ust.hk/) links to [the official quickstart](https://github.com/turingaicloud/quickstart). Its sample sinfo output says `infinite`, illustrating that documentation snapshots differ from today's configuration. Neither its FAQ nor tuxiv.conf guide documents a separate 24-hour QOS or a special submission route.

## Does tcloud automatically grant longer jobs?

The inspected official SDK commit is `fe7a61e9ae9846fa17c54cdd7558ce1d108804fd`.

- [tuxivconfig.go](https://github.com/turingaicloud/tcloud-sdk/blob/fe7a61e9ae9846fa17c54cdd7558ce1d108804fd/cli/tcloudcli/tuxivconfig.go#L146-L176) turns each `job.general` item directly into a `#SBATCH --...` line.
- [tcloudcli.go](https://github.com/turingaicloud/tcloud-sdk/blob/fe7a61e9ae9846fa17c54cdd7558ce1d108804fd/cli/tcloudcli/tcloudcli.go#L458-L488) submits that file through remote `sbatch` under the configured user. Its optional SSH prefix merely adds configured hops; there is no automatic QOS/partition selection or privileged time-limit override in this path.
- Local `tcloud submit --help` advertises only help and verbose flags, not a long-duration service.
- These are source-level findings, not a fresh SDK submission test. The inspected public source was not asserted to be byte-for-byte identical to the installed binary.
- Source snapshots and Git blob IDs are saved in [source evidence](tacc_sdk_time_limit_sources_20261103.json).

A `time=24:00:00` request in tuxiv.conf would request 24 hours; it does not grant permission to exceed MaxTime. No authorized current mechanism for a 24-hour run has been established by this read-only audit. Resolving the difference requires current service-side documentation or an administrator-confirmed allocation/QOS change; no contact was sent.

