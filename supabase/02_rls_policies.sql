-- =====================================================================
-- MPLAD DRISHTI — Phase 1 RLS Policies (Step 1 of Phase 1)
-- Run this SECOND, after 01_schema.sql has succeeded.
-- Enforces access control at the DATABASE level, not just in the
-- frontend — per the explicit principle in the project doc.
-- =====================================================================

-- ---------- helper: read the calling user's own profile ----------
-- SECURITY DEFINER so this function can read `profiles` even though the
-- RLS policy on `profiles` itself would otherwise block a user from
-- seeing rows other than their own — this function is what policies on
-- OTHER tables (mps, districts, etc.) call to find out who's asking.
create or replace function public.current_profile()
returns table (role text, state_id uuid, district_id uuid, mp_id uuid, is_active boolean)
language sql
security definer set search_path = public
stable
as $$
  select role, state_id, district_id, mp_id, is_active
  from profiles
  where id = auth.uid();
$$;

-- ---------- enable RLS on every table ----------
alter table states enable row level security;
alter table districts enable row level security;
alter table mps enable row level security;
alter table profiles enable row level security;

-- ---------- profiles: a user can only ever see/edit their own row ----------
drop policy if exists profiles_select_own on profiles;
create policy profiles_select_own on profiles
  for select using (id = auth.uid());

drop policy if exists profiles_update_own on profiles;
create policy profiles_update_own on profiles
  for update using (id = auth.uid());

-- Note: there is deliberately NO policy allowing a user to change their
-- own role/jurisdiction — that must be done by an admin using the
-- Supabase service-role key (bypasses RLS), not through the app.

-- ---------- states: everyone active can see the full states list ----------
-- (There's no meaningful reason to hide the list of Indian states from
-- any authenticated role — the restriction that matters is on MP/work
-- records, not on this reference table.)
drop policy if exists states_select_active on states;
create policy states_select_active on states
  for select using (
    exists (select 1 from current_profile() p where p.is_active)
  );

-- ---------- districts: scoped by role ----------
drop policy if exists districts_select_scoped on districts;
create policy districts_select_scoped on districts
  for select using (
    exists (
      select 1 from current_profile() p
      where p.is_active and (
        p.role = 'ministry'
        or (p.role = 'state_authority' and p.state_id = districts.state_id)
        or (p.role = 'district_authority' and p.district_id = districts.id)
        or (p.role = 'mp' and p.mp_id in (
              select id from mps where mps.district_id = districts.id
            ))
      )
    )
  );

-- ---------- mps: scoped by role ----------
drop policy if exists mps_select_scoped on mps;
create policy mps_select_scoped on mps
  for select using (
    exists (
      select 1 from current_profile() p
      where p.is_active and (
        p.role = 'ministry'
        or (p.role = 'state_authority' and p.state_id = mps.state_id)
        or (p.role = 'district_authority' and p.district_id = mps.district_id)
        or (p.role = 'mp' and p.mp_id = mps.id)
      )
    )
  );

-- =====================================================================
-- HOW TO VERIFY THIS ACTUALLY WORKS (do this before trusting it):
--
-- 1. In the Supabase SQL Editor, you're running as an elevated role
--    that BYPASSES RLS — so testing there will look like it "works"
--    even if the policies are wrong. That's a false positive trap.
-- 2. Real verification: create a demo user (Step 1c), log in as that
--    user from your actual frontend (using the anon key, not the
--    service key), and confirm the query results are actually scoped.
-- 3. Or: use `select set_config('request.jwt.claims', ...)` tricks in
--    SQL to simulate a specific user — more advanced, do the frontend
--    test first since it's simpler and closer to reality.
-- =====================================================================