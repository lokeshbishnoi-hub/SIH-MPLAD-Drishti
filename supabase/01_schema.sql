-- =====================================================================
-- MPLAD DRISHTI — Phase 1 Schema (Step 1 of Phase 1)
-- Run this FIRST in the Supabase SQL Editor.
-- Covers: states, districts, mps, profiles, auth trigger.
-- Does NOT yet cover: vendors, works, expenditures, alerts,
-- investigations, audit_logs — those come in Step 2, after this is
-- verified working (per the incremental-build principle).
-- =====================================================================

-- ---------- states ----------
create table if not exists states (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  code text
);

-- ---------- districts ----------
-- District names are NOT globally unique (e.g. multiple states have a
-- district with a common name), so uniqueness is scoped to (state_id, name).
create table if not exists districts (
  id uuid primary key default gen_random_uuid(),
  state_id uuid not null references states(id) on delete restrict,
  name text not null,
  unique (state_id, name)
);

-- ---------- mps ----------
-- district_id is nullable: nominated Rajya Sabha members (e.g. those
-- nominated for contributions to art/literature/science) have no
-- constituency/district. This is real, not a data gap to paper over.
create table if not exists mps (
  id uuid primary key default gen_random_uuid(),
  legacy_mp_id text unique,              -- e.g. "MP0001", ties back to the CSV-era ID
  mp_name text not null,
  state_id uuid not null references states(id) on delete restrict,
  district_id uuid references districts(id) on delete set null,
  constituency text,
  house text,                            -- currently "Parliamentary Member" in source data;
                                          -- NOT reliably split into Lok Sabha / Rajya Sabha yet
  entitlement_cr numeric,
  goi_release_cr numeric,
  unreleased_cr numeric,
  tenure text,
  nodal_district text,
  data_source text,                      -- which real CSV this row came from — provenance
  created_at timestamptz not null default now()
);

create index if not exists idx_mps_state on mps(state_id);
create index if not exists idx_mps_district on mps(district_id);

-- ---------- profiles ----------
-- One row per authenticated user, linked 1:1 to Supabase's built-in
-- auth.users table. This is what RLS policies will key off of.
create table if not exists profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text,
  role text not null check (role in ('ministry','state_authority','district_authority','mp')),
  state_id uuid references states(id),        -- required for state_authority
  district_id uuid references districts(id),  -- required for district_authority
  mp_id uuid references mps(id),               -- required for mp role
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- A profile should have exactly the jurisdiction field that matches its role.
-- (Enforced here as a check constraint, not just convention — matches the
-- "don't rely on frontend-only enforcement" principle, applied to data
-- integrity too, not just access control.)
alter table profiles add constraint profile_jurisdiction_matches_role check (
  (role = 'ministry') or
  (role = 'state_authority' and state_id is not null) or
  (role = 'district_authority' and district_id is not null) or
  (role = 'mp' and mp_id is not null)
);

-- ---------- auth trigger ----------
-- When a new user signs up via Supabase Auth, auto-create a placeholder
-- profile row. Role/jurisdiction get set separately (by an admin, via the
-- demo seed step, or a signup flow you build later) — this trigger just
-- guarantees every auth user has a corresponding profiles row so RLS
-- policies always have something to check against.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, full_name, role, is_active)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'full_name', new.email),
    coalesce(new.raw_user_meta_data->>'role', 'mp'),  -- safe default; must be corrected before real access
    false  -- inactive until an admin assigns a real role + jurisdiction
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();